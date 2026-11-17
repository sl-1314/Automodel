# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Expert-parallel (EP) extension of the 0-diff recipe (C9-B/C validated).

Replicates vllm's naive DP-EP semantics on the trainer side so a
``world_size``-replicated (or FSDP2-sharded) model computes bitwise the
same routed-expert outputs as the vllm rollout engine running
TP1 x DP(world) x EP(world):

  local:    gate GEMM + ``ops.topk_softmax`` on this rank's tokens
            (pre-dispatch, identical to the non-EP recipe — C9-A A2 proof)
  dispatch: all-gather [x, topk_weights, topk_ids] in rank order
            (pure data movement)
  compute:  ``moe_align_block_size(ti_g, 64, E GLOBAL, expert_map,
            ignore_invalid=True)`` — only this rank's E/world local experts
            enter the segments; Triton grouped GEMMs (the recipe's BI
            config) on the FULL w13/w2 transposed views with expert ids
            offset to global rows; routing weight applied in GEMM2 (fp32);
            ``ops.moe_sum`` skips non-local slots
  combine:  ``dist.reduce(fused, root=0, SUM)`` + per-rank broadcast of
            slices (mirrors vllm BI reduce(root0) + scatter)

The backward (C9-C, module-level validated vs the non-EP recipe adjoint:
w13 6.9e-5 / w2 2.5e-5 / gate 2.7e-3 / x 1.3e-3 — all bf16 noise level)
runs the same four-step EP adjoint:

  1. adjoint of combine: ``G = allreduce(SUM)`` of dy scattered into the
     ``[T_sum, H]`` layout (fused_total = sum_k fused_k)
  2. differentiable recompute over LOCAL experts only, token-major order,
     multiplier = the saved fp32 ``tw_g`` (exact topk_softmax outputs);
     grads via ``torch.autograd.grad`` with zero-fill so the collective
     adjoints below always execute (determinism: C9-C1 bitwise)
  3. adjoint of dispatch: ``allreduce(SUM)`` of the x_g / tw_g grads
     (each rank holds only its local-expert partial), slice own rows
  4. logits grad: differentiable softmax-select-normalize on the LOCAL
     logits (selection fixed from saved ids), driven by the sliced tw grad

Grads for w13/w2 are full-shaped with nonzero rows only at local experts —
FSDP2's reduce-scatter then sums the complementary rank contributions
(C9-C2: module norm_rel worst 4.5e-3 vs the no-FSDP EP reference; the
residual is FSDP2's bf16 mean accumulation, production-faithful).

Expert placement is linear: rank k owns globals ``[k*base, (k+1)*base)``.

``install_zero_diff_ep`` must run AFTER ``install_zero_diff`` (it replaces
the per-layer mlp forward installed there). ``wrap_fsdp2_ep`` applies the
C9-B/C-validated FSDP2 call-graph grouping — see its docstring for why the
groups must follow the recipe's actual call graph, not the module tree.
"""

import types

import torch

# ---------------------------------------------------------------------------
# EP state (process-global; one EP group per process, set at install time)
# ---------------------------------------------------------------------------
_EP = None


def init_ep(rank, world, num_experts=256):
    global _EP
    assert num_experts % world == 0, (num_experts, world)
    base = num_experts // world
    start = rank * base
    em = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    em[start:start + base] = torch.arange(base, dtype=torch.int32, device="cuda")
    _EP = {"rank": rank, "world": world, "start": start, "base": base,
           "num_experts": num_experts, "expert_map": em}
    return _EP


def _gather(x, tw, ti):
    """Dispatch: all-gather [x, tw, ti] in rank order (data-only)."""
    import torch.distributed as dist

    ep = _EP
    dev = x.device
    T = x.shape[0]
    t_loc = torch.tensor([T], dtype=torch.int64, device=dev)
    sizes_t = [torch.empty(1, dtype=torch.int64, device=dev) for _ in range(ep["world"])]
    dist.all_gather(sizes_t, t_loc)
    sizes = [int(t.item()) for t in sizes_t]
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    T_sum = sum(sizes)

    x_g = torch.empty(T_sum, x.shape[1], dtype=x.dtype, device=dev)
    tw_g = torch.empty(T_sum, tw.shape[1], dtype=tw.dtype, device=dev)
    ti_g = torch.empty(T_sum, ti.shape[1], dtype=ti.dtype, device=dev)
    r, off = ep["rank"], offsets[ep["rank"]]
    x_g[off:off + T].copy_(x)
    tw_g[off:off + T].copy_(tw)
    ti_g[off:off + T].copy_(ti)
    for src in range(ep["world"]):
        lo, hi = offsets[src], offsets[src] + sizes[src]
        if lo == hi:
            continue
        dist.broadcast(x_g[lo:hi], src=src)
        dist.broadcast(tw_g[lo:hi], src=src)
        dist.broadcast(ti_g[lo:hi], src=src)
    return x_g, tw_g, ti_g, sizes, offsets


def _combine_scatter(fused, sizes, offsets):
    """Combine: reduce(root=0, SUM) then data-only scatter of each rank's rows.

    Mirrors vllm's BI path (ncclReduce root=0 + root scatter). Every rank
    participates in every broadcast (its own slice lands in ``out``, other
    ranks' slices use ``fused`` rows as scratch).
    """
    import torch.distributed as dist

    ep = _EP
    rk = ep["rank"]
    dist.reduce(fused, dst=0, op=dist.ReduceOp.SUM)
    lo, hi = offsets[rk], offsets[rk] + sizes[rk]
    out = torch.empty(hi - lo, fused.shape[1], dtype=fused.dtype, device=fused.device)
    if rk == 0 and lo < hi:
        out.copy_(fused[lo:hi])
    for src in range(ep["world"]):
        slo, shi = offsets[src], offsets[src] + sizes[src]
        if slo == shi:
            continue  # all ranks hold identical sizes -> consistent skip
        buf = out if src == rk else fused[slo:shi]
        dist.broadcast(buf, src=0)
    return out


# ---------------------------------------------------------------------------
# routed-experts EP path with autograd
# ---------------------------------------------------------------------------
_BI_CONFIG = {
    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8, "SPLIT_K": 1,
}


class MoERoutedEP(torch.autograd.Function):
    """EP routed-experts forward (bitwise B2/C-validated sequence) + adjoint."""

    @staticmethod
    def forward(ctx, x, logits, w13, w2, E, N, TOPK):
        import triton.language as tl  # noqa: F401
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            invoke_fused_moe_triton_kernel,
        )
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
        from vllm.model_executor.layers.fused_moe.activation import (
            MoEActivation,
            apply_moe_activation,
        )

        ep = _EP
        dev = x.device
        T, H = x.shape
        N2 = 2 * N

        # local topk (pre-dispatch; same kernel call as the recipe's MoERouted)
        tw = torch.empty(T, TOPK, dtype=torch.float32, device=dev)
        ti = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
        tei = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
        is_padding = torch.zeros(T, dtype=torch.bool, device=dev)
        ops.topk_softmax(tw, ti, tei, logits, True, None, is_padding)

        # dispatch
        x_g, tw_g, ti_g, sizes, offsets = _gather(x, tw, ti)
        T_sum = x_g.shape[0]

        # expert assignment: local experts only (expert_map maps global->local,
        # non-local -> -1; ignore_invalid drops those slots from the segments)
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            ti_g, _BI_CONFIG["BLOCK_SIZE_M"], E, ep["expert_map"], True, True
        )
        # kernel indexes GLOBAL rows of the full weight views: offset local->global
        eids_g = torch.where(
            expert_ids >= 0, expert_ids + ep["start"], expert_ids
        ).to(torch.int32)

        # GEMM1 + activation + GEMM2 (identical kernel/config to the recipe)
        c1 = torch.empty(T_sum, TOPK, N2, dtype=torch.bfloat16, device=dev)
        invoke_fused_moe_triton_kernel(
            x_g, w13, c1, None, None, None, sorted_token_ids, eids_g,
            num_tokens_post_padded, False, TOPK, _BI_CONFIG,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
            use_int4_w4a16=False, per_channel_quant=False,
        )
        c2 = torch.empty(T_sum * TOPK, N, dtype=torch.bfloat16, device=dev)
        apply_moe_activation(MoEActivation.SILU, c2, c1.view(-1, N2))
        c3 = torch.empty(T_sum, TOPK, H, dtype=torch.bfloat16, device=dev)
        invoke_fused_moe_triton_kernel(
            c2, w2, c3, None, None, tw_g, sorted_token_ids, eids_g,
            num_tokens_post_padded, True, 1, _BI_CONFIG,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
            use_int4_w4a16=False, per_channel_quant=False,
        )

        # local top-k reduction, expert_map-aware (skips non-local slots)
        fused = torch.empty(T_sum, H, dtype=torch.bfloat16, device=dev)
        ops.moe_sum(c3, fused, ti_g, ep["expert_map"])

        # combine
        out = _combine_scatter(fused, sizes, offsets)

        ctx.save_for_backward(x, logits, w13, w2, x_g, ti_g, tw_g)
        ctx.meta = (
            E, N, TOPK, tuple(sizes), tuple(offsets),
            ep["rank"], ep["world"], ep["start"], ep["base"],
        )
        return out

    @staticmethod
    def backward(ctx, dy):
        import torch.distributed as dist

        x, logits, w13, w2, x_g, ti_g, tw_g = ctx.saved_tensors
        E, N, TOPK, sizes, offsets, rank, world, start, base = ctx.meta
        dev = x.device
        lo, hi = offsets[rank], offsets[rank] + sizes[rank]

        # 1) adjoint of combine (collective: must run on every rank)
        G = torch.zeros(x_g.shape[0], x_g.shape[1], dtype=torch.float32, device=dev)
        G[lo:hi] = dy.float()
        dist.all_reduce(G, op=dist.ReduceOp.SUM)

        # 2) differentiable recompute over LOCAL experts (token-major order ->
        #    partition-independent summation -> bitwise-deterministic reruns)
        import torch.nn.functional as F

        x_gl = x_g.detach().requires_grad_(x.requires_grad)
        tw_gl = tw_g.detach().requires_grad_(True)
        w13l = w13.detach().requires_grad_(w13.requires_grad)
        w2l = w2.detach().requires_grad_(w2.requires_grad)
        til = ti_g.long()
        out = torch.zeros(x_g.shape[0], TOPK, x_g.shape[1],
                          dtype=torch.float32, device=dev)
        with torch.enable_grad():
            for e in range(start, start + base):
                rows = til == e
                if not rows.any():
                    continue
                tok, slot = rows.nonzero(as_tuple=True)
                xs = x_gl[tok]
                c1 = torch.matmul(xs, w13l[e].t())
                c2 = F.silu(c1[:, :N]) * c1[:, N:]
                c3 = torch.matmul(c2, w2l[e].t())
                out[tok, slot] = c3.float()
            routed_k = (out * tw_gl[..., None]).sum(1)
            leaves = [x_gl, tw_gl, w13l, w2l]
            active = [l for l in leaves if l.requires_grad]
            gs = [None] * 4
            if active:
                grads = torch.autograd.grad(routed_k, active, G,
                                            allow_unused=True)
                gmap = {id(l): g for l, g in zip(active, grads)}
                gs = [gmap.get(id(l)) for l in leaves]
        gx_g, gtw_g, gw13, gw2 = gs
        # zero-fill so the dispatch-adjoint collectives always run
        if gx_g is None:
            gx_g = torch.zeros_like(x_g)
        if gtw_g is None:
            gtw_g = torch.zeros_like(tw_g)
        if gw13 is None:
            gw13 = torch.zeros_like(w13)
        if gw2 is None:
            gw2 = torch.zeros_like(w2)

        # 3) adjoint of dispatch: sum rank partials, slice own rows
        gx32 = gx_g.to(torch.float32)
        dist.all_reduce(gx32, op=dist.ReduceOp.SUM)
        grad_x = gx32[lo:hi].to(x.dtype)
        gtw = gtw_g.contiguous()
        dist.all_reduce(gtw, op=dist.ReduceOp.SUM)
        grad_tw_own = gtw[lo:hi]

        # 4) logits grad via the differentiable softmax path (local rows)
        grad_logits = None
        if ctx.needs_input_grad[1]:
            with torch.enable_grad():
                ll = logits.detach().requires_grad_(True)
                p = torch.softmax(ll.float(), dim=-1)
                sel = p.gather(1, ti_g[lo:hi].long())
                twl = sel / sel.sum(-1, keepdim=True)
                gl, = torch.autograd.grad(
                    twl, ll, grad_outputs=grad_tw_own.to(twl.dtype),
                    allow_unused=True,
                )
            grad_logits = gl.to(logits.dtype) if gl is not None \
                else torch.zeros_like(logits)

        return grad_x, grad_logits, gw13, gw2, None, None, None


def moe_zero_diff_forward_ep(self, hidden_states, padding_mask=None, cp_mesh=None):
    """The recipe's moe_zero_diff_forward with the routed part as
    MoERoutedEP.apply (grad-enabled). Shared expert + gate + add unchanged."""
    from .zero_diff import BILinear, SiluMul

    assert padding_mask is None, "0-diff recipe requires a full (unpadded) sequence"
    was_3d = hidden_states.ndim == 3
    x = hidden_states.reshape(-1, hidden_states.shape[-1])

    w13 = self.experts.gate_and_up_projs.transpose(1, 2)  # [E, 2N, H] view
    w2 = self.experts.down_projs.transpose(1, 2)  # [E, H, N] view
    E = self.experts.n_routed_experts
    N = w13.shape[1] // 2
    TOPK = self.gate.topk

    logits = BILinear.apply(x, self.gate.weight)
    routed = MoERoutedEP.apply(x, logits, w13, w2, E, N, TOPK)

    w_gu = torch.cat(
        [self.shared_experts.gate_proj.weight, self.shared_experts.up_proj.weight],
        dim=0,
    )
    gu = BILinear.apply(x, w_gu)
    sa = SiluMul.apply(gu)
    sd = BILinear.apply(sa, self.shared_experts.down_proj.weight)
    sg = BILinear.apply(x, self.shared_expert_gate.weight)
    shared_out = torch.sigmoid(sg) * sd

    out = routed + shared_out
    if was_3d:
        out = out.reshape(hidden_states.shape)
    return out


# ---------------------------------------------------------------------------
# installers
# ---------------------------------------------------------------------------
def install_zero_diff_ep(model, ep_rank, ep_world):
    """Patch the per-layer mlp forwards to the EP path (after
    ``install_zero_diff``). Returns the install count dict."""
    backend = model.model.language_model
    first = next(iter(backend.layers.values()))
    init_ep(ep_rank, ep_world, first.mlp.experts.n_routed_experts)

    n_moe = 0
    for layer in backend.layers.values():
        mlp = layer.mlp
        mlp.forward = types.MethodType(moe_zero_diff_forward_ep, mlp)
        n_moe += 1
    return dict(moe_ep=n_moe)


def wrap_fsdp2_ep(model):
    """Apply the C9-B/C-validated FSDP2 grouping to a zero-diff model.

    The groups must follow the recipe backend's ACTUAL call graph, not the
    module tree: ``text_backend_zero_diff_forward`` never enters
    ``layer.__call__`` — it calls ``layer.mlp(...)`` / ``layer.linear_attn``
    / ``layer.self_attn(...)`` as module calls and reads norm weights
    directly. A whole-layer group's pre-forward unshard therefore never
    fires and sharded norm weights hit the flashinfer cutlass kernel
    (cudaErrorIllegalAddress — C9-B2 v3). Validated grouping (v4):

      per layer:  ``fully_shard(gdn._fp32_params)`` FIRST (child-first;
                  mixed-dtype group raises otherwise), then
                  ``fully_shard(gdn)`` / ``fully_shard(self_attn)``,
                  then ``fully_shard(mlp)``
      root-ish:   ``fully_shard(language_model)`` (embed_tokens + norms —
                  stays materialized through the backend forward;
                  ``self.language_model(...)`` IS a real module call),
                  ``fully_shard(model.model)`` (root),
                  ``fully_shard(model.lm_head)`` (sibling root; the recipe's
                  chunked lm_head GEMM summons it)

    GDN gates (fp32 ``_fp32_params``) additionally need a summon window at
    the gdn forward entry: the recipe reads ``self.A_log/dt_bias`` directly
    and never calls ``_fp32_params.forward``, so the gate group's hooks
    never fire on their own (C9-B1: sharded [8]-of-[32] tensors reaching
    the fla kernel abort). ``unshard()`` must run AFTER the gdn's own
    pre-forward lazy init (calling it before the first gdn forward breaks
    the FSDP2 root-state order — v6).

    Vision params are frozen first: the text-only forward never touches
    them, and FSDP grad sync hangs if a wrapped param produces no grad.
    """
    from torch.distributed.fsdp import fully_shard as _fs

    n_frozen = 0
    for name, p in model.named_parameters():
        if ".visual." in name or name.startswith("model.visual"):
            p.requires_grad_(False)
            n_frozen += 1

    backend = model.model.language_model
    for key, layer in backend.layers.items():
        if hasattr(layer, "linear_attn"):
            _fs(layer.linear_attn._fp32_params)  # child-first: fp32 gate
            _fs(layer.linear_attn)
        else:
            _fs(layer.self_attn)
        _fs(layer.mlp)
    _fs(backend)      # embed_tokens + all layernorms + leftovers
    _fs(model.model)  # root (vision etc.)
    _fs(model.lm_head)  # sibling root; grads reduce-scattered (mean)

    # GDN fp32 gate summon windows (wrap the INSTALLED forward)
    n_summon = 0
    for key, layer in backend.layers.items():
        if hasattr(layer, "linear_attn"):
            gdn = layer.linear_attn

            def _make_summon(fwd):
                def _summon_fwd(self, *args, **kw):
                    self._fp32_params.unshard()  # summon window (B1 pattern)
                    return fwd(*args, **kw)
                return _summon_fwd

            gdn.forward = types.MethodType(_make_summon(gdn.forward), gdn)
            n_summon += 1
    return dict(fsdp2_ep_wrapped=True, frozen_visual_tensors=n_frozen,
                gdn_summon_windows=n_summon)
