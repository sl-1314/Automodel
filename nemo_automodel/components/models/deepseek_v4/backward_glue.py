"""C6 backward glue (stage 1: tail chain + MoE + mhc boundaries).

Per the zero-mismatch skill (`03-reuse-backward.md` / `02-qat-and-ste.md`):
every module sits behind autograd-opaque Triton/TileLang/JIT kernels, so each
one gets an `autograd.Function` whose **forward calls the original kernel with
the original arguments** (the value path is unchanged — bitwise identical,
including under `no_grad`, where the Function degenerates to the bare kernel
call) and whose **backward computes the VJP from quantities saved by that
forward**, either as a closed form or by recomputing the module's torch
reference under `enable_grad` and taking `autograd.grad` (the generic bridge).

Recompute precision is fp32 (the DS4 precedent: backward is closed-form
PyTorch + glue, not kernel reimplementation); gradients are cast back to each
input's dtype at the Function boundary.  Discrete quantities (top-k expert
ids, clamp masks) are FIXED from the forward — never recomputed in backward
(IDs are discrete states, not quantization; the continuous scores around them
stay differentiable).

Verdicts per module live in `outputs/.../c6/inventory.md` (S0–S11).
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _sqrt_softplus_open(x: torch.Tensor) -> torch.Tensor:
    # open-coded log1p branch (matches the kernel's non-log1p path):
    # sp = max(x, 0) + log1p(exp(-|x|)); phi = sqrt(sp)
    z = torch.exp(-torch.abs(x))
    sp = torch.clamp(x, min=0.0) + torch.log1p(z)
    return torch.sqrt(sp)


def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float,
                  keep_fp32: bool = False) -> torch.Tensor:
    """RMSNorm reference: y = x * rsqrt(mean(x^2) + eps) * w, fp32 math,
    output cast to x's dtype.  keep_fp32=True skips the cast (backward
    bridges: a bf16->fp32 upcast's backward DOWNCASTS the grad to bf16, so
    every intermediate cast quantizes the gradient path — the recompute
    graph keeps everything fp32 and casts only at the Function boundary)."""
    x32 = x.float()
    r = torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
    y = x32 * r * weight.float()
    return y if keep_fp32 else y.to(x.dtype)


def _mhc_pre_ref(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                 hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                 keep_fp32: bool = False):
    """Torch reference of the mhc_pre math (mhc.py `_mhc_pre_torch`).
    keep_fp32=True keeps layer_input in fp32 (backward bridges — see
    _rms_norm_ref: intermediate bf16 casts quantize the grad path)."""
    s, n, h = residual.shape
    dtype = residual.dtype
    x_flat = residual.reshape(s, n * h).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x_flat, fn) * rsqrt
    pre_raw = mixes[:, :n]
    post_raw = mixes[:, n:2 * n]
    comb_raw = mixes[:, 2 * n:].reshape(s, n, n)
    pre_base = hc_base[:n]
    post_base = hc_base[n:2 * n]
    comb_base = hc_base[2 * n:].reshape(n, n)
    pre = torch.sigmoid(pre_raw * hc_scale[0] + pre_base) + hc_pre_eps
    post = hc_post_mult_value * torch.sigmoid(
        post_raw * hc_scale[1] + post_base)
    comb = comb_raw * hc_scale[2] + comb_base
    comb = comb.softmax(-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=1)
    if not keep_fp32:
        layer_input = layer_input.to(dtype)
    return post.unsqueeze(-1), comb, layer_input


def _mhc_post_ref(x, residual, post_layer_mix, comb_res_mix,
                  keep_fp32: bool = False):
    """Torch reference of the mhc_post math (mhc.py `_mhc_post_torch`)."""
    if post_layer_mix.ndim == 2:
        post_layer_mix = post_layer_mix.unsqueeze(-1)
    out = post_layer_mix * x.float().unsqueeze(1) + (
        comb_res_mix.unsqueeze(-1) * residual.float().unsqueeze(2)
    ).sum(dim=1)
    return out if keep_fp32 else out.type_as(x)


def _silu_clamp_ref(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    """silu_and_mul_clamp math (silu_and_mul_masked_post_quant.cuh:41):
    gate upper-clamped to L only, up clamped to [-L, L]; silu(g)*u in fp32,
    cast back to the input dtype.  gate = first half, up = second half."""
    d = gate_up.shape[-1] // 2
    g = gate_up[..., :d].float()
    u = gate_up[..., d:].float()
    g = torch.clamp(g, max=limit)
    u = torch.clamp(u, min=-limit, max=limit)
    h = F.silu(g) * u
    return h.to(gate_up.dtype)


# ---------------------------------------------------------------------------
# S0: hc_broadcast — trivial VJP dX = dY.sum(dim=1)


class HcBroadcastFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, hc_mult):
        from sglang.kernels.ops.layernorm.mhc import hc_broadcast
        ctx.save_for_backward(x)
        ctx.hc_mult = hc_mult
        return hc_broadcast(x, hc_mult)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        # sum in fp32, then round to the input dtype (mixed-precision contract)
        return grad_out.float().sum(dim=1).to(x.dtype), None


# ---------------------------------------------------------------------------
# S1: mhc_pre (L0) — recompute bridge over _mhc_pre_ref + RMSNorm


class MhcPreFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, residual, fn, hc_scale, hc_base, norm_weight,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                sinkhorn_repeat, norm_eps):
        from sglang.kernels.ops.layernorm.mhc import mhc_pre
        ctx.save_for_backward(residual, fn, hc_scale, hc_base, norm_weight)
        ctx.rms_eps = rms_eps
        ctx.hc_pre_eps = hc_pre_eps
        ctx.hc_sinkhorn_eps = hc_sinkhorn_eps
        ctx.hc_post_mult_value = hc_post_mult_value
        ctx.sinkhorn_repeat = sinkhorn_repeat
        ctx.norm_eps = norm_eps
        return mhc_pre(
            residual=residual, fn=fn, hc_scale=hc_scale, hc_base=hc_base,
            rms_eps=rms_eps, hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat, norm_weight=norm_weight,
            norm_eps=norm_eps,
        )

    @staticmethod
    def backward(ctx, grad_post, grad_comb, grad_y):
        residual, fn, hc_scale, hc_base, norm_weight = ctx.saved_tensors
        leaves = [t.detach().requires_grad_(True)
                  for t in (residual, fn, hc_scale, hc_base, norm_weight)]
        r_l, fn_l, sc_l, b_l, w_l = leaves
        with torch.enable_grad():
            # fp32 recompute: upcast the bf16 input ONCE (its grad is
            # quantized to bf16 only at the Function boundary below); the
            # chain itself stays fp32 — intermediate bf16 casts would
            # quantize the grad path at every upcast node.
            r32 = r_l.float() if r_l.dtype != torch.float32 else r_l
            post, comb, li = _mhc_pre_ref(
                r32, fn_l, sc_l, b_l, ctx.rms_eps, ctx.hc_pre_eps,
                ctx.hc_sinkhorn_eps, ctx.hc_post_mult_value,
                ctx.sinkhorn_repeat, keep_fp32=True)
            y = _rms_norm_ref(li, w_l, ctx.norm_eps, keep_fp32=True)
            grads = torch.autograd.grad(
                (post, comb, y), leaves,
                (grad_post, grad_comb, grad_y), allow_unused=True)
        return tuple(g.to(t.dtype) if g is not None else None
                     for g, t in zip(grads, (residual, fn, hc_scale,
                                             hc_base, norm_weight))) + (
            None, None, None, None, None, None)


# ---------------------------------------------------------------------------
# S2: mhc boundary (non-L0) — recompute bridge over
#     _mhc_post_ref -> _mhc_pre_ref -> RMSNorm (the unfused reference of
#     mhc_fused_post_pre; fn enters the pre-norm GEMM as F.linear(x, fn))


class BoundaryFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, value, streams, post, comb, fn, hc_scale, hc_base,
                norm_weight, hc_mult, rms_eps, hc_eps, hc_post_mult_value,
                sinkhorn_iters, norm_eps):
        from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc \
            import apply_mhc_post_pre_boundary
        ctx.save_for_backward(value, streams, post, comb, fn, hc_scale,
                              hc_base, norm_weight)
        ctx.hc_mult = hc_mult
        ctx.rms_eps = rms_eps
        ctx.hc_eps = hc_eps
        ctx.hc_post_mult_value = hc_post_mult_value
        ctx.sinkhorn_iters = sinkhorn_iters
        ctx.norm_eps = norm_eps
        return apply_mhc_post_pre_boundary(
            value, streams, post, comb, fn, hc_scale, hc_base, hc_mult,
            rms_eps, hc_eps, hc_post_mult_value, sinkhorn_iters, norm_weight,
            norm_eps, fn_transpose=False)

    @staticmethod
    def backward(ctx, grad_streams, grad_x, grad_post, grad_comb, _flag):
        value, streams, post, comb, fn, hc_scale, hc_base, norm_weight = \
            ctx.saved_tensors
        tensors = (value, streams, post, comb, fn, hc_scale, hc_base,
                   norm_weight)
        leaves = [t.detach().requires_grad_(True) for t in tensors]
        (x_l, s_l, p_l, c_l, fn_l, sc_l, b_l, w_l) = leaves
        with torch.enable_grad():
            # fp32 recompute (see MhcPreFn.backward): upcast bf16 inputs
            # once, no intermediate casts — the value path's bf16 casts
            # would quantize the grad path at every upcast node.
            x32 = x_l.float() if x_l.dtype != torch.float32 else x_l
            s32 = s_l.float() if s_l.dtype != torch.float32 else s_l
            r_cur = _mhc_post_ref(x32, s32, p_l, c_l, keep_fp32=True)
            post_out, comb_out, li = _mhc_pre_ref(
                r_cur, fn_l, sc_l, b_l, ctx.rms_eps, ctx.hc_eps,
                ctx.hc_eps, ctx.hc_post_mult_value, ctx.sinkhorn_iters,
                keep_fp32=True)
            y = _rms_norm_ref(li, w_l, ctx.norm_eps, keep_fp32=True)
            grads = torch.autograd.grad(
                (r_cur, y, post_out.squeeze(-1), comb_out), leaves,
                (grad_streams, grad_x, grad_post, grad_comb),
                allow_unused=True)
        return tuple(g.to(t.dtype) if g is not None else None
                     for g, t in zip(grads, tensors)) + (
            None, None, None, None, None, None, None)


# ---------------------------------------------------------------------------
# S3: mhc_post (final read) — closed-form linear VJP


class MhcPostFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, residual, post, comb):
        from sglang.kernels.ops.layernorm.mhc import mhc_post
        ctx.save_for_backward(x, residual, post, comb)
        return mhc_post(x, residual, post, comb)

    @staticmethod
    def backward(ctx, grad_out):
        x, residual, post, comb = ctx.saved_tensors
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(True)
                      for t in (x, residual, post, comb)]
            x_l, s_l, p_l, c_l = leaves
            out = _mhc_post_ref(x_l, s_l, p_l, c_l)
            grads = torch.autograd.grad(out, leaves, grad_out)
        return tuple(g.to(t.dtype) for g, t in zip(
            grads, (x, residual, post, comb)))


# ---------------------------------------------------------------------------
# S4: fused_hc_head — recompute bridge (linear + sigmoid gate + weighted sum)


class HcHeadFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
        from sglang.kernels.ops.layernorm.mhc_head import fused_hc_head
        ctx.save_for_backward(x, hc_fn, hc_scale, hc_base)
        ctx.norm_eps = norm_eps
        ctx.hc_eps = hc_eps
        return fused_hc_head(x, hc_fn, hc_scale, hc_base,
                             norm_eps, hc_eps)

    @staticmethod
    def backward(ctx, grad_y):
        x, hc_fn, hc_scale, hc_base = ctx.saved_tensors
        tensors = (x, hc_fn, hc_scale, hc_base)
        leaves = [t.detach().requires_grad_(True) for t in tensors]
        x_l, fn_l, sc_l, b_l = leaves
        with torch.enable_grad():
            s, n, h = x.shape
            # kernel math (mhc_head.py torch reference): the rms stats are
            # over the flattened (n·h) row, not per-stream.
            x32 = x_l.float().reshape(s, n * h)
            r = torch.rsqrt(x32.square().mean(-1, keepdim=True)
                            + ctx.norm_eps)
            mixes = F.linear(x32, fn_l) * r
            pre = torch.sigmoid(mixes * sc_l[0] + b_l) + ctx.hc_eps
            y = (pre.unsqueeze(-1) * x32.reshape(s, n, h)).sum(1).to(x.dtype)
            grads = torch.autograd.grad(y, leaves, grad_y,
                                        allow_unused=True)
        return tuple(g.to(t.dtype) if g is not None else None
                     for g, t in zip(grads, tensors)) + (None, None)


# ---------------------------------------------------------------------------
# S5: rms_norm BI sites — closed-form VJP


class RmsNormFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, eps):
        from sglang.srt.batch_invariant_ops import rms_norm_batch_invariant
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return rms_norm_batch_invariant(x, weight, eps)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        x32 = x.float()
        w32 = weight.float()
        r = torch.rsqrt(x32.square().mean(-1, keepdim=True) + ctx.eps)
        go = grad_out.float()
        # dX = r*(go*w) - x * r^3 * mean(go*w*x)
        gw = go * w32
        dX = r * gw - x32 * (r * r * r) * (gw * x32).mean(
            -1, keepdim=True)
        # dW = sum over rows of go * (x * r)
        dW = (go * (x32 * r)).reshape(-1, x32.shape[-1]).sum(0) \
            if x.dim() > 2 else (go * (x32 * r)).sum(0)
        return dX.to(x.dtype), dW.to(weight.dtype), None


# ---------------------------------------------------------------------------
# S6: linear_bf16_fp32 — standard GEMM VJP (fp32 accumulate)


class LinearBF16FP32Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w):
        from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32
        ctx.save_for_backward(x, w)
        return linear_bf16_fp32(x, w)

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        go = grad_out.float()
        dX = go @ w.float()
        dW = go.t() @ x.float()
        return dX.to(x.dtype), dW.to(w.dtype)


# ---------------------------------------------------------------------------
# S8: silu_and_mul_clamp — closed-form VJP with clamp masks


class SiluClampFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, gate_up, limit):
        from sglang.kernels.ops.activation.silu_and_mul_clamp import (
            silu_and_mul_clamp,
        )
        ctx.save_for_backward(gate_up)
        ctx.limit = limit
        d = gate_up.shape[-1] // 2
        out = gate_up.new_empty(gate_up.shape[:-1] + (d,))
        silu_and_mul_clamp(gate_up, out, limit)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        gate_up, = ctx.saved_tensors
        limit = ctx.limit
        d = gate_up.shape[-1] // 2
        g = gate_up[..., :d].float()
        u = gate_up[..., d:].float()
        # clamp masks: gate upper-clamped to L only; up clamped to [-L, L]
        # (inclusive at the boundary — torch clamp semantics)
        g_mask = (g <= limit).to(g.dtype)
        u_mask = ((u >= -limit) & (u <= limit)).to(u.dtype)
        g_c = torch.clamp(g, max=limit)
        u_c = torch.clamp(u, min=-limit, max=limit)
        sig = torch.sigmoid(g_c)
        silu = g_c * sig
        dsilu = sig * (1.0 + g_c * (1.0 - sig))
        go = grad_out.float()
        # h = silu(g_c) * u_c: dg carries the u_c factor, du the silu factor
        dg = go * u_c * dsilu * g_mask
        du = go * silu * u_mask
        return torch.cat([dg, du], dim=-1).to(gate_up.dtype), None


# ---------------------------------------------------------------------------
# S9: router gates — discrete-fixed IDs + continuous weight recompute
#     hash: w_k = phi(l[t, e_k]) / (sum_k phi + 1e-20); ids from tid2eid
#     fused gate: same weights; ids from the (activated + bias) ranking


class HashTopkFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, input_ids, tid2eid, num_fused_shared_experts,
                routed_scaling_factor, scoring_func):
        from sglang.kernels.ops.moe.dsv4 import hash_topk
        w, ids = hash_topk(logits, input_ids, tid2eid,
                           num_fused_shared_experts, routed_scaling_factor,
                           scoring_func)
        ctx.save_for_backward(logits, ids)
        ctx.renorm_eps = 1e-20
        ctx.num_fused_shared_experts = num_fused_shared_experts
        return w, ids

    @staticmethod
    def backward(ctx, grad_w, grad_ids):
        logits, ids = ctx.saved_tensors
        # shared slots (k >= K_routed) carry the constant 1/rsf -> no grad
        K = ids.shape[1] - ctx.num_fused_shared_experts
        grad_w = grad_w.clone()
        if ctx.num_fused_shared_experts:
            grad_w[..., K:] = 0.0
        sel = ids[..., :K].long() if ctx.num_fused_shared_experts \
            else ids.long()
        with torch.enable_grad():
            l_l = logits.detach().requires_grad_(True)
            ph = _sqrt_softplus_open(l_l.gather(1, sel))
            denom = ph.sum(-1, keepdim=True) + ctx.renorm_eps
            w_ref = ph / denom
            g = torch.autograd.grad(w_ref, l_l, grad_w)[0]
        return g, None, None, None, None, None


class GateFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, bias, topk, scoring_func,
                num_fused_shared_experts, renormalize, routed_scaling_factor,
                apply_routed_scaling_factor_on_output):
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
        w, ids = moe_fused_gate(
            logits, bias, topk, scoring_func=scoring_func,
            num_fused_shared_experts=num_fused_shared_experts,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=(
                apply_routed_scaling_factor_on_output))
        ctx.save_for_backward(logits, ids)
        ctx.renorm_eps = 0.0  # renormalize uses the >0 guard, no epsilon
        ctx.num_fused_shared_experts = num_fused_shared_experts
        return w, ids

    @staticmethod
    def backward(ctx, grad_w, grad_ids):
        logits, ids = ctx.saved_tensors
        K = ids.shape[1] - ctx.num_fused_shared_experts
        grad_w = grad_w.clone()
        if ctx.num_fused_shared_experts:
            grad_w[..., K:] = 0.0
        sel = ids[..., :K].long() if ctx.num_fused_shared_experts \
            else ids.long()
        with torch.enable_grad():
            l_l = logits.detach().requires_grad_(True)
            ph = _sqrt_softplus_open(l_l.gather(1, sel))
            denom = ph.sum(-1, keepdim=True)
            denom = torch.where(denom > 0.0, denom,
                                torch.ones_like(denom))
            w_ref = ph / denom
            g = torch.autograd.grad(w_ref, l_l, grad_w)[0]
        # bias gets NO gradient: it enters only the (discrete) ranking
        return g, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# S10: fused_experts (+ moe_sum_reduce rsf) — recompute with discrete states


class FusedExpertsFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w13, w2, topk_weights, topk_ids, router_logits, mrc):
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        ctx.save_for_backward(x, w13, w2, topk_weights, topk_ids)
        ctx.rsf = mrc.routed_scaling_factor
        ctx.limit = mrc.swiglu_limit if mrc.swiglu_limit is not None else 10.0
        topk_output = StandardTopKOutput(topk_weights, topk_ids,
                                         router_logits.detach())
        # fused_experts is in-place on hidden_states — feed a clone
        return fused_experts(x.clone(), w13, w2, topk_output, mrc)

    @staticmethod
    def backward(ctx, grad_out):
        x, w13, w2, w, ids = ctx.saved_tensors
        rsf = ctx.rsf
        limit = ctx.limit
        tensors = (x, w13, w2, w)
        leaves = [t.detach().requires_grad_(True) for t in tensors]
        x_l, w13_l, w2_l, w_l = leaves
        sel = ids.long()  # [T, K] discrete states, fixed
        with torch.enable_grad():
            # expert-grouped recompute: the per-slot gather w13[sel]
            # would materialize [T, K, 2I, H] (~85 GB at the proxy
            # shapes); grouping by expert keeps every GEMM at
            # [m, H] @ [H, 2I] with the weights as VIEWS of the
            # leaves.  Per token the top-k experts are distinct, so
            # each (t, k) slot lands in exactly one expert group —
            # the same sum, grouped accumulation order (backward does
            # not need bitwise, only the fp32 noise floor).  The chain
            # is fp32-smooth (no intermediate bf16 casts — see the
            # S1/boundary VJP notes in c6_vjp_checks).
            T, E = x.shape[0], w13.shape[0]
            routed = x_l.new_zeros((T, x.shape[-1]), dtype=torch.float32)
            for e in range(E):
                hit = (sel == e)
                if not bool(hit.any()):
                    continue
                t_idx, k_idx = hit.nonzero(as_tuple=True)
                h_e = x_l[t_idx].float() @ w13_l[e].float().t()  # [m,2I]
                act_e = _silu_clamp_ref(h_e, limit)              # [m,I]
                y_e = act_e @ w2_l[e].float().t()                # [m,H]
                w_e = w_l[t_idx, k_idx].float()                  # [m]
                routed = routed.index_add(
                    0, t_idx, rsf * (y_e * w_e.unsqueeze(-1)))
            grads = torch.autograd.grad(
                routed, leaves, grad_out.float(), allow_unused=True)
        out = []
        for g, t in zip(grads, tensors):
            out.append(None if g is None else g.to(t.dtype))
        # pad for topk_ids / router_logits / mrc (non-tensor grad targets)
        return tuple(out) + (None, None, None)


# ---------------------------------------------------------------------------
# C6 stage 2 — attention middle (A1 norm_rope family).  The Functions are
# math-validated per module (VJP checks); the aligned.py wiring lands when
# the whole attention middle is grad-transparent (flash_mla A3 unblocks the
# flow — wiring norm_rope alone would not change the c64 probe).

_FREQS_DOC = """freqs_cis is complex64 [max_pos, rope_dim/2]; the kernels
read the REAL view (re, im interleaved) so pair j = elements (2j, 2j+1)
rotated by (cos_j, sin_j) = freqs_real[pos, 2j:2j+2] (verified against
main_norm_rope.cuh fused_q_norm_rope and rope.cuh deepseek_rope_kernel)."""


def _rope_freqs(freqs_cis: torch.Tensor, positions: torch.Tensor):
    """Gather per-token (cos, sin) pairs [T, rope_dim/2] each, fp32."""
    fr = torch.view_as_real(freqs_cis.detach()).flatten(-2)  # [P, R] (re,im)
    f = fr[positions.long()]                                  # [T, R]
    return f[..., 0::2], f[..., 1::2]                         # cos, sin


def _rotate_pairs(x_pairs: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                  inverse: bool):
    """x_pairs [..., R/2, 2] (real, imag per pair) -> rotated, fp32 in/out.

    forward:  out = (x_r*c - x_i*s, x_r*s + x_i*c)
    inverse:  out = (x_r*c + x_i*s, x_i*c - x_r*s)   (conjugate multiply)
    """
    xr, xi = x_pairs[..., 0], x_pairs[..., 1]
    if inverse:
        return torch.stack((xr * cos + xi * sin, xi * cos - xr * sin), -1)
    return torch.stack((xr * cos - xi * sin, xr * sin + xi * cos), -1)


class QNormRopeFn(torch.autograd.Function):
    """A1a: fused_q_norm_rope — per-head WEIGHTLESS RMSNorm (q_norm is
    folded as the norm-self scale; the q path passes eps only), then RoPE
    on the last rope_dim elements.  Forward = the kernel (bitwise value
    path); backward = closed-form VJP recomputed fp32-smooth from the
    saved q (norm-self VJP + inverse rotation).  freqs layout: see _FREQS_DOC"""

    @staticmethod
    def forward(ctx, q, eps, freqs_cis, positions):
        from sglang.kernels.ops.attention.dsv4.norm_rope import (
            fused_q_norm_rope,
        )
        q_out = torch.empty_like(q)
        fused_q_norm_rope(q, q_out, eps, freqs_cis, positions)
        ctx.save_for_backward(q, positions)
        ctx.eps = eps
        ctx.freqs_cis = freqs_cis.detach()
        return q_out

    @staticmethod
    def backward(ctx, grad_out):
        q, positions = ctx.saved_tensors
        D = q.shape[-1]
        with torch.enable_grad():
            q_l = q.detach().requires_grad_(True)
            q32 = q_l.float()                                  # [T,H,D]
            r = torch.rsqrt(q32.square().mean(-1, keepdim=True)
                            + ctx.eps)
            xn = q32 * r
            cos, sin = _rope_freqs(ctx.freqs_cis, positions)   # [T,R/2]
            cos = cos.view(-1, *([1] * (q.dim() - 2)), cos.shape[-1])
            sin = sin.view(-1, *([1] * (q.dim() - 2)), sin.shape[-1])
            tail = xn[..., D - cos.shape[-1] * 2:] \
                .unflatten(-1, (cos.shape[-1], 2))
            rot = _rotate_pairs(tail, cos, sin, inverse=False)
            out = torch.cat(
                [xn[..., :D - cos.shape[-1] * 2],
                 rot.flatten(-2)], dim=-1)
            g = torch.autograd.grad(out, (q_l,), grad_out.float())[0]
        return g.to(q.dtype), None, None, None


class RopeInplaceFn(torch.autograd.Function):
    """A1c: fused_rope_inplace (the O-side inverse rope on the tail slice).
    The map is a pure rotation (linear, orthogonal): the VJP is the
    OPPOSITE-orientation rotation of grad_out — no saved tensors needed.
    Forward calls the kernel on a clone (the aligned value path replaces
    the in-place slice mutation with `o = cat([o[..., :-R], out], -1)`,
    values identical).  freqs layout: see _FREQS_DOC"""

    @staticmethod
    def forward(ctx, tail, freqs_cis, positions, inverse):
        from sglang.kernels.ops.attention.dsv4.norm_rope import (
            fused_rope_inplace,
        )
        out = tail.clone()
        fused_rope_inplace(out, None, freqs_cis, positions, inverse)
        ctx.save_for_backward(positions)
        ctx.freqs_cis = freqs_cis.detach()
        ctx.inverse = inverse
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (positions,) = ctx.saved_tensors
        cos, sin = _rope_freqs(ctx.freqs_cis, positions)       # [T,R/2]
        # broadcast [T,1,...,R/2] over the head dims
        cos = cos.view(cos.shape[0], *([1] * (grad_out.dim() - 2)),
                       cos.shape[-1])
        sin = sin.view(sin.shape[0], *([1] * (grad_out.dim() - 2)),
                       sin.shape[-1])
        g = grad_out.float().unflatten(-1, (cos.shape[-1], 2))
        # VJP of a rotation = conjugate rotation; forward used `inverse`,
        # so backward uses the opposite orientation.
        dx = _rotate_pairs(g, cos, sin, inverse=not ctx.inverse)
        return dx.flatten(-2).to(grad_out.dtype), None, None, None


# ===================== A3: flash_mla recompute bridge =====================
#
# V4 paged row layout (kv_layout.cuh traits, cross-checked against the pool
# byte widths): kDataBytes=576, kScaleBytes=8, kBytesPerToken=584,
# kPageAlign=576; page row = ceil(page_size*584/576)*576 bytes.
#   data row of slot s (page-relative): [s*576, s*576+576)
#     bytes [0:448)   fp8 e4m3, 7 groups of 64 (one ue8m0 scale each)
#     bytes [448:576) 64 bf16 roped dims (not quantized)
#   scale row of slot s: [page_size*576 + s*8, +8) — 7 ue8m0 exponent bytes
#     (scale = 2^(byte-127)) + 1 pad byte.
# Index-table entries are TOKEN LOCS: page = loc // page_size,
# slot = loc % page_size.


_A3_DEBUG = None  # test hook: dict to stash backward intermediates


def _v4_decode_rows(pool, locs, page_size):
    """Decode V4 paged rows to fp32 K rows [L, 512] (exact: fp8 -> fp32 and
    bf16 -> fp32 are value-exact, scales are powers of two)."""
    u8 = pool if pool.dtype == torch.uint8 else pool.view(torch.uint8)
    locs = locs.to(u8.device).long()
    page, slot = locs // page_size, locs % page_size
    cols = torch.arange(576, device=u8.device)
    data = u8[page.unsqueeze(1), (slot * 576).unsqueeze(1) + cols]
    scale = u8[page.unsqueeze(1),
               (page_size * 576 + slot * 8).unsqueeze(1)
               + torch.arange(8, device=u8.device)]
    k_fp8 = data[:, :448].view(torch.float8_e4m3fn).float()
    k_rope = data[:, 448:576].view(torch.bfloat16).float()
    e = (scale[:, :7].view(torch.uint8).int() - 127).float()
    k_nope = (k_fp8.view(-1, 7, 64) * torch.pow(2.0, e).unsqueeze(-1)) \
        .flatten(1)
    return torch.cat([k_nope, k_rope], dim=1)


class FlashMlaFn(torch.autograd.Function):
    """A3: ``flash_mla_with_kvcache`` with an fp32 recompute backward.

    Forward = the exact engine kernel call (opaque wheel kernel; the value
    path is bitwise — the closure ``spec["kernel_call"]`` reproduces the
    engine call site).  Backward recomputes the sparse attention in fp32:

      k(loc)  = V4 decode (see _v4_decode_rows)
      s       = softmax_scale * q.k   per head
      Z       = sum_j exp(s_j - m) + exp(sink_h - m)   -- the per-head
                attn_sink is a VIRTUAL SINK LOGIT: it adds exp(sink) to the
                softmax denominator (sink token carries no value row);
                empirically pinned vs the wheel kernel on the c64 replay
                (all 5 layers, residual = bf16 output rounding only)
      o       = p @ k   (V = the same decoded row; o's last 64 dims are
                roped — the caller applies the inverse rope, A1c)

    Gradients (fp32-smooth): dQ (closed form), dSink, and dK handed to the
    K-store / compressor shadow latents through the case-B STE (identity on
    the dequantized values).  The index tables are DISCRETE STATES saved
    from forward, never recomputed.  Attended locs with no shadow row
    (constants — rows written by earlier steps) contribute to the
    recompute VALUES but receive no gradient.

    Shadow-row mapping: ``row_locs_swa``/``row_locs_extra`` give the pool
    loc each shadow row was written to this step; ``spec`` carries
    ``kernel_call(q) -> o`` plus {"sm_scale", "page_swa", "page_extra"}.
    """

    @staticmethod
    def forward(ctx, q, sink, shadow_swa, shadow_extra, pool_swa,
                pool_extra, row_locs_swa, row_locs_extra, swa_idx, swa_len,
                extra_idx, extra_len, spec):
        o = spec["kernel_call"](q)
        ctx.save_for_backward(q, sink, pool_swa, pool_extra,
                              row_locs_swa, row_locs_extra,
                              swa_idx, swa_len, extra_idx, extra_len)
        ctx.spec = spec
        ctx.shadow_dtypes = (
            None if shadow_swa is None else shadow_swa.dtype,
            None if shadow_extra is None else shadow_extra.dtype,
        )
        return o

    @staticmethod
    def backward(ctx, do):
        (q, sink, pool_swa, pool_extra, row_locs_swa, row_locs_extra,
         swa_idx, swa_len, extra_idx, extra_len) = ctx.saved_tensors
        sm = ctx.spec["sm_scale"]
        page_s = ctx.spec["page_swa"]
        page_e = ctx.spec["page_extra"]
        T, H, D = q.shape
        dev = q.device
        do32 = do.detach().float()
        q32 = q.detach().float()
        sink32 = sink.detach().float()

        def pad_table(idx, lens):
            idx = idx.to(dev)
            valid = ((torch.arange(idx.shape[1], device=dev).unsqueeze(0)
                      < lens.to(dev).unsqueeze(1)) & (idx >= 0))
            return idx.clamp(min=0).long(), valid

        loc_s, val_s = pad_table(swa_idx, swa_len)
        if extra_idx is not None and extra_idx.shape[-1] > 0:
            loc_e, val_e = pad_table(extra_idx, extra_len)
        else:
            loc_e = torch.zeros(T, 0, device=dev, dtype=torch.long)
            val_e = torch.zeros(T, 0, device=dev, dtype=torch.bool)

        def pool_state(pool, locs2d, valid, page):
            """decode rows + per-(t,slot) decode-row index (or -1)."""
            if pool is None or int(valid.sum()) == 0:
                # 1-row dummy when the table has slots but none valid, so
                # the clamped gather below stays in bounds (masked anyway)
                return (torch.zeros(1 if locs2d.numel() else 0, D,
                                    device=dev),
                        torch.full(locs2d.shape, -1, device=dev,
                                   dtype=torch.long))
            uloc, inv = torch.unique(locs2d[valid], return_inverse=True)
            K_u = _v4_decode_rows(pool, uloc, page)
            row = torch.full(locs2d.shape, -1, device=dev, dtype=torch.long)
            row[valid] = inv
            return K_u, row

        K_s, row_s = pool_state(pool_swa, loc_s, val_s, page_s)
        K_e, row_e = pool_state(pool_extra, loc_e, val_e, page_e)
        # gather [T, L, D]; invalid lanes point at decode row 0 and are
        # masked out of the softmax below.
        Kg = torch.cat([K_s[row_s.clamp(min=0)],
                        K_e[row_e.clamp(min=0)]], dim=1)
        val = torch.cat([val_s, val_e], dim=1)

        s = sm * torch.einsum("thd,tld->thl", q32, Kg)
        s = s.masked_fill(~val.unsqueeze(1), float("-inf"))
        m = s.amax(-1, keepdim=True)                           # [T,H,1]
        e = torch.exp(s - m)                                  # 0 on invalid
        Eterm = torch.exp(sink32.view(1, H) - m.squeeze(-1))  # [T,H]
        Z = e.sum(-1) + Eterm                                 # [T,H]
        p = e / Z.unsqueeze(-1)                               # [T,H,L]

        dP = torch.einsum("thd,tld->thl", do32, Kg)           # [T,H,L]
        beta = (p * dP).sum(-1)                               # [T,H]
        ds = p * (dP - beta.unsqueeze(-1))                    # 0 on invalid
        dq = sm * torch.einsum("thl,tld->thd", ds, Kg)        # [T,H,D]
        o_rc = torch.einsum("thl,tld->thd", p, Kg)            # [T,H,D]
        dsink = -((Eterm / Z) * (do32 * o_rc).sum(-1)).sum(0)  # [H]
        # dK has TWO paths (K is both the value rows AND the score keys):
        # the V path p^T do plus the score path sm * ds^T q.
        dk = (torch.einsum("thl,thd->tld", p, do32)
              + sm * torch.einsum("thl,thd->tld", ds, q32))  # [T,L,D]

        def grad_rows(pool, locs2d, valid, page, row, row_locs, dk_full):
            """dK in decode-row space -> shadow-row space (index_add)."""
            if pool is None or int(valid.sum()) == 0 \
                    or row_locs is None:
                return None
            uloc, _ = torch.unique(locs2d[valid], return_inverse=True)
            rl = row_locs.to(dev).long()
            order = torch.argsort(rl)
            slocs = rl[order]
            pos = torch.searchsorted(slocs, uloc)
            pos_c = pos.clamp(max=slocs.shape[0] - 1)
            hit = slocs[pos_c] == uloc
            dec2sh = torch.where(hit, order[pos_c],
                                 torch.full_like(order[pos_c], -1))
            sh = dec2sh[row]                                  # [T,L] or -1
            m2 = valid & (sh >= 0)
            if int(m2.sum()) == 0:
                return None
            g = torch.zeros(rl.shape[0], D, device=dev)
            return g.index_add_(0, sh[m2], dk_full[m2])

        g_s = grad_rows(pool_swa, loc_s, val_s, page_s, row_s,
                        row_locs_swa, dk[:, :loc_s.shape[1]])
        g_e = grad_rows(pool_extra, loc_e, val_e, page_e, row_e,
                        row_locs_extra, dk[:, loc_s.shape[1]:])

        if _A3_DEBUG is not None:
            _A3_DEBUG["g_s"] = None if g_s is None else g_s.detach()
            _A3_DEBUG["g_e"] = None if g_e is None else g_e.detach()
            _A3_DEBUG["p"] = p.detach()
            _A3_DEBUG["ds"] = ds.detach()
        dt_s, dt_e = ctx.shadow_dtypes
        grads = [dq.to(q.dtype), dsink.to(sink.dtype),
                 None if g_s is None else g_s.to(dt_s),
                 None if g_e is None else g_e.to(dt_e)]
        grads.extend([None] * 9)  # pools / locs / tables / spec
        return tuple(grads)
