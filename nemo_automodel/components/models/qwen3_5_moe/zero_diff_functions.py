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

"""Autograd.Function wrappers restoring the gradient graph through the
0-diff vllm-replica forward for Qwen3.5-MoE.

Port of the validated C6 build (task records:
``outputs/2026-11-10-qwen35-0diff/c6/`` in the task workspace). The forward
of every Function is the exact C5 vllm leaf sequence (bitwise 0-diff vs the
vllm engine; gate receipt ``c5/c5_gate.json``); backward is a same-math
differentiable recompute, tolerance-accepted per the C6 backward inventory.

Conventions:
 - forward = the exact C5 leaf sequence (vllm BI kernels; bitwise)
 - backward = same-math differentiable recompute (torch / fla / sdpa);
   only quantities saved by the Function forward are used
 - in-place kernels (flashinfer gemma norms) run on clones in forward
 - discrete choices (MoE top-k ids) are fixed from saved ids (P3)

All vllm / flashinfer / fla imports are deferred to the call site so this
module imports without those dependencies present.
"""

import os

import torch
import torch.nn.functional as F

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")


def _recompute_grads(ref, dy, leaves):
    """torch.autograd.grad on a recompute reference, tolerating non-grad
    leaves (frozen parameters, e.g. experts frozen for the C8 training step):
    only the leaves that require grad are differentiated; the others were
    constants in the reference graph and get None back (same contract as
    ``ctx.needs_input_grad`` in BILinear.backward)."""
    active = [l for l in leaves if l.requires_grad]
    if not active:
        return [None] * len(leaves)
    grads = torch.autograd.grad(ref, active, grad_outputs=dy.to(ref.dtype))
    it = iter(grads)
    return [next(it) if l.requires_grad else None for l in leaves]


# ----------------------------------------------------------------------------
# module 1: BI linear GEMM
# ----------------------------------------------------------------------------
class BILinear(torch.autograd.Function):
    """y = x @ w^T. Forward: vllm BI Triton MM (bitwise). Backward: cuBLAS."""

    @staticmethod
    def forward(ctx, x, w):
        from vllm.model_executor.determinism.batch_invariant import (
            linear_batch_invariant,
        )

        ctx.save_for_backward(x, w)
        return linear_batch_invariant(x, w)

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        dy = dy.to(x.dtype)
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = torch.matmul(dy, w).to(x.dtype)
        if ctx.needs_input_grad[1]:
            dw = torch.matmul(dy.transpose(0, 1), x).to(w.dtype)
        return dx, dw


# ----------------------------------------------------------------------------
# module 2: Gemma RMS norms (flashinfer, in-place -> clones in forward)
# ----------------------------------------------------------------------------
def _gemma_bwd_math(x, residual, w, dy, eps):
    """Shared fp32 backward math. s = x (+ residual); y = (1+w) * s * r(s).

    Returns (ds, dw) with ds = dL/dx (== dL/dresidual when residual given).
    """
    s = x.float() if residual is None else x.float() + residual.float()
    d = s.shape[-1]
    var = s.pow(2).mean(-1, keepdim=True)
    r = torch.rsqrt(var + eps)
    n = s * r
    w1p = w.float() + 1.0
    dyf = dy.float()
    dn = dyf * w1p
    ds = r * (dn - n * (dn * n).mean(-1, keepdim=True))
    dw = (dyf * n).reshape(-1, d).sum(0)
    return ds, dw


class GemmaRMSNormFn(torch.autograd.Function):
    """Layer-0 norm (no residual). Forward: flashinfer gemma_rmsnorm."""

    @staticmethod
    def forward(ctx, x, w, eps):
        from flashinfer.norm import gemma_rmsnorm

        xc = x.clone()
        out = gemma_rmsnorm(xc, w, eps)
        ctx.save_for_backward(x, w)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        ds, dw = _gemma_bwd_math(x, None, w, dy, ctx.eps)
        return ds.to(x.dtype), dw.to(w.dtype), None


class GemmaAddRMSNormFn(torch.autograd.Function):
    """residual-deferring Gemma fused add+norm.

    Forward: clones x and residual, then flashinfer gemma_fused_add_rmsnorm
    in-place on the clones (kernel computes y from the UNROUNDED fp32 sum —
    that is the C5 root fix vs the bf16-added residual; the clone protects
    the autograd-visible input). Returns (y, residual_new).
    """

    @staticmethod
    def forward(ctx, x, residual, w, eps):
        from flashinfer.norm import gemma_fused_add_rmsnorm

        xc = x.clone()
        rc = residual.clone()
        gemma_fused_add_rmsnorm(xc, rc, w, eps)
        ctx.save_for_backward(x, residual, w)
        ctx.eps = eps
        return xc, rc

    @staticmethod
    def backward(ctx, dy, dres):
        x, residual, w = ctx.saved_tensors
        ds, dw = _gemma_bwd_math(x, residual, w, dy, ctx.eps)
        # dres (grad of the returned new-residual) flows to both x and the
        # ORIGINAL residual the same way: the stored value is round(x+res)
        dtotal = ds + dres.float()
        return dtotal.to(x.dtype), dtotal.to(residual.dtype), dw.to(w.dtype), None


# ----------------------------------------------------------------------------
# module 3: GDN core (conv -> prep -> chunk delta rule -> gated norm)
# ----------------------------------------------------------------------------
def gdn_ref(mixed_qkvz, ba, conv_w, A_log, dt_bias, w_norm, dims):
    """Differentiable same-math reference of the GDN core (for backward).

    dims: dict(num_k_heads, num_v_heads, head_k_dim, head_v_dim,
               conv_kernel, eps)
    """
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    H = dims["num_k_heads"]
    HV = dims["num_v_heads"]
    K = dims["head_k_dim"]
    V = dims["head_v_dim"]
    KC = dims["conv_kernel"]
    eps = dims["eps"]
    T = mixed_qkvz.shape[0]
    key_dim, value_dim = H * K, HV * V
    conv_dim = 2 * key_dim + value_dim

    mixed_qkv, z = mixed_qkvz.split([conv_dim, value_dim], dim=-1)
    b, a = ba.split([HV, HV], dim=-1)

    # causal conv + silu (depthwise, left pad KC-1) — same math as
    # vllm causal_conv1d_fn / HF torch fallback
    xw = mixed_qkv.t().unsqueeze(0)  # [1, conv_dim, T]
    xw = F.pad(xw, (KC - 1, 0))
    conv = F.conv1d(xw, conv_w.view(conv_w.shape[0], 1, KC), groups=conv_w.shape[0])
    conv = F.silu(conv)[0].t().contiguous()  # [T, conv_dim]

    # split (prep layout: q | k | v along conv_dim); q/k stay UNnormalized —
    # fla's in-kernel l2norm (eps 1e-6, same as fused_post_conv_prep) does it
    q = conv[:, :key_dim].reshape(T, H, K).contiguous()
    k = conv[:, key_dim : 2 * key_dim].reshape(T, H, K).contiguous()
    v = conv[:, 2 * key_dim :].reshape(T, HV, V).contiguous()
    g = -torch.exp(A_log.float()).unsqueeze(0) * F.softplus(a.float() + dt_bias.float())
    beta = torch.sigmoid(b.float())

    # chunk gated delta rule (fla Triton, autograd; GVA-native q at H heads,
    # default scale 1/sqrt(K), l2norm in-kernel, beta post-sigmoid, g log-space)
    o = chunk_gated_delta_rule(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        use_qk_l2norm_in_kernel=True,
    )
    if isinstance(o, tuple):  # fla returns (output, final_state)
        o = o[0]
    core = o.squeeze(0)  # [T, HV, V]

    # gated rms norm (HF Qwen3_5MoeRMSNormGated math, fp32)
    coref = core.float()
    var = coref.pow(2).mean(-1, keepdim=True)
    normed = coref * torch.rsqrt(var + eps) * w_norm.float()
    out = normed * F.silu(z.float().view(T, HV, V))
    return out.to(mixed_qkvz.dtype).reshape(T, value_dim)


class GDNCore(torch.autograd.Function):
    """Forward: the C5 GDN leaf sequence steps 2-5 (bitwise). Backward:
    autograd through gdn_ref (torch conv + fla chunk + torch norm)."""

    @staticmethod
    def forward(ctx, mixed_qkvz, ba, conv_w, A_log, dt_bias, w_norm, dims):
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_fn,
        )
        from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (
            fused_post_conv_prep,
        )
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            fi_chunk_gated_delta_rule,
        )
        from vllm.models.qwen3_5.nvidia.ops.fused_gdn_gated_norm import (
            fused_gdn_gated_rms_norm,
        )

        H, HV = dims["num_k_heads"], dims["num_v_heads"]
        K, V = dims["head_k_dim"], dims["head_v_dim"]
        KC, eps = dims["conv_kernel"], dims["eps"]
        dev = mixed_qkvz.device
        T = mixed_qkvz.shape[0]
        key_dim, value_dim = H * K, HV * V
        conv_dim = 2 * key_dim + value_dim

        mixed_qkv, _z = mixed_qkvz.split([conv_dim, value_dim], dim=-1)
        b, a = ba.split([HV, HV], dim=-1)

        # cache slot 0 is NULL_BLOCK_ID: use a nonzero line so the kernel
        # actually computes this sequence (C5 finding)
        conv_state = torch.zeros(2, conv_dim, KC - 1, dtype=torch.bfloat16, device=dev)
        conv_out = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            conv_w,
            None,
            activation="silu",
            conv_states=conv_state,
            has_initial_state=torch.zeros(1, dtype=torch.bool, device=dev),
            cache_indices=torch.tensor([1], dtype=torch.int32, device=dev),
            query_start_loc=torch.tensor([0, T], dtype=torch.int32, device=dev),
        ).transpose(0, 1)

        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=conv_out.contiguous(),
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            num_k_heads=H,
            head_k_dim=K,
            head_v_dim=V,
            apply_l2norm=True,
            output_g_exp=True,
        )

        initial_state = torch.zeros(1, HV, V, K, dtype=torch.float32, device=dev)
        o, _ = fi_chunk_gated_delta_rule(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=torch.tensor([0, T], dtype=torch.int32, device=dev),
            use_qk_l2norm_in_kernel=False,
        )
        core = o.squeeze(0)

        z2d = mixed_qkvz[:, conv_dim:]
        out2d = core.reshape(T, -1)
        fused_gdn_gated_rms_norm(
            x=out2d,
            z=z2d,
            weight=w_norm,
            out=out2d,
            eps=eps,
            num_v_heads=HV,
            head_v_dim=V,
            norm_before_gate=True,
            activation="silu",
        )
        ctx.save_for_backward(mixed_qkvz, ba, conv_w, A_log, dt_bias, w_norm)
        ctx.dims = dims
        return out2d

    @staticmethod
    def backward(ctx, dy):
        mixed_qkvz, ba, conv_w, A_log, dt_bias, w_norm = ctx.saved_tensors
        dims = ctx.dims
        leaves = [
            t.detach().requires_grad_(t.requires_grad)
            for t in (mixed_qkvz, ba, conv_w, A_log, dt_bias, w_norm)
        ]
        with torch.enable_grad():
            ref = gdn_ref(*leaves, dims)
            grads = _recompute_grads(ref, dy, leaves)
        return tuple(grads) + (None,)


# ----------------------------------------------------------------------------
# module 4: MoE routed experts
# ----------------------------------------------------------------------------
class MoERouted(torch.autograd.Function):
    """Forward: the C5 MoE routed leaf sequence (topk_softmax -> align ->
    fused w13/w2 GEMMs -> moe_sum; bitwise). Backward: torch per-expert
    recompute with the saved topk ids (discrete routing fixed, P3) and the
    routing weights recomputed differentiably from logits.

    w13 [E, 2N, K] / w2 [E, N_out, K_in] may be transposed views of the
    NeMo-layout expert parameters (gate_and_up_projs / down_projs): the
    Triton kernel indexes B through explicit strides.
    """

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

        T, H = x.shape
        dev = x.device
        N2 = 2 * N
        bi_config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "SPLIT_K": 1,
        }
        tw = torch.empty(T, TOPK, dtype=torch.float32, device=dev)
        ti = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
        tei = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
        is_padding = torch.zeros(T, dtype=torch.bool, device=dev)
        ops.topk_softmax(tw, ti, tei, logits, True, None, is_padding)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            ti, bi_config["BLOCK_SIZE_M"], E, None, True
        )
        c1 = torch.empty(T, TOPK, N2, dtype=torch.bfloat16, device=dev)
        invoke_fused_moe_triton_kernel(
            x, w13, c1, None, None, None, sorted_token_ids, expert_ids,
            num_tokens_post_padded, False, TOPK, bi_config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
            use_int4_w4a16=False, per_channel_quant=False,
        )
        c2 = torch.empty(T * TOPK, N, dtype=torch.bfloat16, device=dev)
        apply_moe_activation(MoEActivation.SILU, c2, c1.view(-1, N2))
        c3 = torch.empty(T, TOPK, H, dtype=torch.bfloat16, device=dev)
        invoke_fused_moe_triton_kernel(
            c2, w2, c3, None, None, tw, sorted_token_ids, expert_ids,
            num_tokens_post_padded, True, 1, bi_config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
            use_int4_w4a16=False, per_channel_quant=False,
        )
        routed = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
        ops.moe_sum(c3, routed)
        ctx.save_for_backward(x, logits, w13, w2, ti, tw)
        ctx.meta = (E, N, TOPK)
        return routed

    @staticmethod
    def backward(ctx, dy):
        x, logits, w13, w2, ti, tw = ctx.saved_tensors
        E, N, TOPK = ctx.meta
        T, H = x.shape
        til = ti.long()

        leaves = [t.detach().requires_grad_(t.requires_grad) for t in (x, logits, w13, w2)]
        xl, ll, w13l, w2l = leaves
        with torch.enable_grad():
            # routing weights, differentiable wrt logits (selection fixed)
            p = torch.softmax(ll.float(), dim=-1)  # [T, E]
            sel = p.gather(1, til)  # [T, TOPK]
            twl = sel / sel.sum(-1, keepdim=True)

            # per-expert torch recompute of the routed path
            out = torch.zeros(T, TOPK, H, dtype=torch.float32, device=x.device)
            for e in range(E):
                rows = (til == e)
                if not rows.any():
                    continue
                tok, slot = rows.nonzero(as_tuple=True)
                xs = xl[tok]  # [n, H]
                c1 = torch.matmul(xs, w13l[e].t())  # [n, 2N]
                c2 = F.silu(c1[:, :N]) * c1[:, N:]
                c3 = torch.matmul(c2, w2l[e].t())  # [n, H]
                out[tok, slot] = c3.float()
            routed = (out * twl[..., None]).sum(1)
            grads = _recompute_grads(routed, dy, leaves)
        return tuple(grads) + (None, None, None)


class SiluMul(torch.autograd.Function):
    """silu_and_mul: forward = vllm apply_moe_activation (bitwise),
    backward = torch silu*mul."""

    @staticmethod
    def forward(ctx, gu):
        from vllm.model_executor.layers.fused_moe.activation import (
            MoEActivation,
            apply_moe_activation,
        )

        N = gu.shape[-1] // 2
        out = torch.empty(gu.shape[0], N, dtype=gu.dtype, device=gu.device)
        apply_moe_activation(MoEActivation.SILU, out, gu)
        ctx.save_for_backward(gu)
        return out

    @staticmethod
    def backward(ctx, dy):
        (gu,) = ctx.saved_tensors
        N = gu.shape[-1] // 2
        leaf = gu.detach().requires_grad_(gu.requires_grad)
        with torch.enable_grad():
            ref = F.silu(leaf[..., :N]) * leaf[..., N:]
            g = torch.autograd.grad(ref, leaf, grad_outputs=dy.to(ref.dtype))[0]
        return g.to(gu.dtype)


# ----------------------------------------------------------------------------
# module 5: FA core (qk norm + rope + gate -> paged FA3 -> sigmoid gate mul)
# ----------------------------------------------------------------------------
def fa_ref(q_gate, k_in, v, q_w, k_w, cos_sin, head_dims):
    """Differentiable same-math reference of the FA core (for backward).

    head_dims: dict(num_q_heads, num_kv_heads, head_dim, rotary_dim, eps)
    NeoX pairing (i, i+rotary_dim/2) per the Triton kernel; mrope sections
    collapse to plain positions (text-only, all planes equal).
    """
    nq, kvh, D, RD = (
        head_dims["num_q_heads"],
        head_dims["num_kv_heads"],
        head_dims["head_dim"],
        head_dims["rotary_dim"],
    )
    eps = head_dims["eps"]
    T = q_gate.shape[0]
    half = RD // 2

    qg = q_gate.view(T, nq, 2, D)
    qh, gate = qg[..., 0, :], qg[..., 1, :]
    kh = k_in.view(T, kvh, D)
    vh = v.view(T, kvh, D)

    def _norm(h, w):
        hf = h.float()
        var = hf.pow(2).mean(-1, keepdim=True)
        r = torch.rsqrt(var + eps)
        return (hf * r * (w.float() + 1.0)).to(h.dtype)

    qn = _norm(qh, q_w)
    kn = _norm(kh, k_w)

    cache = cos_sin[:T].float()  # rows 0..T-1 == arange positions
    # [T, 1, half]: broadcast over heads
    cos, sin = cache[:, None, :half], cache[:, None, half:RD]

    def _rope(h):
        x1, x2 = h[..., :half], h[..., half:RD]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat([o1, o2, h[..., RD:]], dim=-1)

    qr = _rope(qn).float()
    kr = _rope(kn).float()
    vr = vh.float()

    # attention: plain causal sdpa-equivalent math (einsum, fp32) — kept
    # independent of the sdpa path used at model scale. GQA: repeat kv heads
    rep = nq // kvh
    qT = qr.permute(1, 0, 2)  # [nq, T, D]
    kT = kr.repeat_interleave(rep, dim=1).permute(1, 2, 0)  # [nq, D, T]
    scores = torch.matmul(qT, kT) * (D ** -0.5)  # [nq, T, T]
    causal = torch.ones(T, T, dtype=torch.bool, device=q_gate.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    vT = vr.repeat_interleave(rep, dim=1).permute(1, 0, 2)  # [nq, T, D]
    oT = torch.matmul(attn, vT)  # [nq, T, D]
    o = oT.permute(1, 0, 2)  # [T, nq, D]

    out = torch.sigmoid(gate.float()) * o
    return out.to(q_gate.dtype).reshape(T, nq * D)


class FACore(torch.autograd.Function):
    """Forward: the C5 FA leaf sequence steps 2-4 (bitwise). Backward:
    autograd through fa_ref (torch norms/rope + full attention math)."""

    @staticmethod
    def forward(ctx, q_gate, k_in, v, q_w, k_w, cos_sin, head_dims, T):
        from vllm.model_executor.layers.fused_qk_norm_rope import (
            fused_qk_rmsnorm_rope_gate,
        )
        from vllm.model_executor.models.qwen3_next import fused_sigmoid_gate_mul
        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
            reshape_and_cache_flash,
        )

        nq, kvh, D, RD = (
            head_dims["num_q_heads"],
            head_dims["num_kv_heads"],
            head_dims["head_dim"],
            head_dims["rotary_dim"],
        )
        eps = head_dims["eps"]
        dev = q_gate.device

        ar = torch.arange(T, dtype=torch.int64, device=dev)
        positions = torch.stack([ar, ar, ar])
        q, k, gate = fused_qk_rmsnorm_rope_gate(
            q_gate,
            k_in,
            q_w,
            k_w,
            cos_sin,
            positions,
            eps,
            nq,
            kvh,
            D,
            RD,
            mrope_section=head_dims["mrope_section"],
            norm_beta=1.0,
        )

        num_blocks = (T + head_dims["kv_block"] - 1) // head_dims["kv_block"] + 1
        kv_cache = torch.zeros(
            num_blocks, kvh, head_dims["kv_block"], 2 * D, dtype=torch.bfloat16, device=dev
        )
        key_cache, value_cache = kv_cache.transpose(1, 2).split(D, dim=-1)
        slot_mapping = torch.arange(T, dtype=torch.int64, device=dev)
        k3 = k.view(T, kvh, D).contiguous()
        v3 = v.view(T, kvh, D).contiguous()
        scale_one = torch.tensor(1.0, device=dev)
        reshape_and_cache_flash(
            k3, v3, key_cache, value_cache, slot_mapping, "auto", scale_one, scale_one
        )
        q3 = q.view(T, nq, D).contiguous()
        out = torch.empty(T, nq, D, dtype=torch.bfloat16, device=dev)
        block_table = torch.zeros(1, 4, dtype=torch.int32, device=dev)
        flash_attn_varlen_func(
            q=q3,
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=torch.tensor([0, T], dtype=torch.int32, device=dev),
            max_seqlen_q=T,
            seqused_k=torch.tensor([T], dtype=torch.int32, device=dev),
            max_seqlen_k=head_dims["max_seqlen_k"],
            softmax_scale=D ** -0.5,
            causal=True,
            alibi_slopes=None,
            window_size=[-1, -1],
            softcap=0,
            block_table=block_table,
            scheduler_metadata=None,
            fa_version=3,
            num_splits=1,
        )
        attn_output = fused_sigmoid_gate_mul(out.reshape(T, nq * D), gate)
        ctx.save_for_backward(q_gate, k_in, v, q_w, k_w, cos_sin)
        ctx.head_dims = head_dims
        return attn_output

    @staticmethod
    def backward(ctx, dy):
        q_gate, k_in, v, q_w, k_w, cos_sin = ctx.saved_tensors
        head_dims = ctx.head_dims
        leaves = [
            t.detach().requires_grad_(t.requires_grad)
            for t in (q_gate, k_in, v, q_w, k_w)
        ]
        with torch.enable_grad():
            ref = fa_ref(*leaves, cos_sin, head_dims)
            grads = _recompute_grads(ref, dy, leaves)
        return tuple(g if g is not None else None for g in grads) + (None, None, None)


# ----------------------------------------------------------------------------
# module 6: training-side logprobs (0-diff scoring kernel stays for the gate;
# the loss path uses fp32 torch log_softmax on the same bf16 logits)
# ----------------------------------------------------------------------------
def logprobs_loss(logits, token_ids):
    """logits [T, V] bf16, token_ids [T] -> fp32 logprobs of token_ids
    (same math as the Triton scoring kernel up to rounding)."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    return lp.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
