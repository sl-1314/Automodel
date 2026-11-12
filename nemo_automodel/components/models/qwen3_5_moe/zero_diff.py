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

"""0-diff (bitwise logprob) recipe installer for the Qwen3.5-MoE AutoModel.

Wires the validated C5/C6 vllm-replica compute path (see
``zero_diff_functions.py``) into a loaded
``Qwen3_5MoeForConditionalGeneration`` so the trainer-side forward is
bitwise-identical to the vllm prefill engine while keeping the autograd graph.

Layout/semantic mapping from the NeMo component to the C5 (HF/vllm) layout:

======================  ==========================  =============================
piece                   NeMo component              C5 recipe expects
======================  ==========================  =============================
GDN in_proj_qkv|z|b|a   separate linears            cat[qkv|z], cat[b|a] per call
GDN A_log/dt_bias       ``_fp32_params`` fp32       A_log fp32 / dt_bias bf16
                                                  (bf16 values; cast is exact)
MoE routed experts      ``gate_and_up_projs``       ``w13`` [E, 2N, H] — passed as
                        [E, H, 2N], ``down_projs``  a transposed VIEW (the Triton
                        [E, N, H]                   kernel indexes B by strides)
MoE router              ``gate.weight`` [E, H]      same
shared expert           ``shared_experts``          cat[gate|up] per call
FA q/k/v                separate q|gate (2x), k, v  cat per call (live graph)
norms                   ``Qwen3NextRMSNorm``        Gemma (1+w) — same formula,
                        (1+w) formula, bf16 params  flashinfer kernel
lm_head                 NeMo linear                 chunked BI GEMM (1024 rows)
======================  ==========================  =============================

Constraints (asserted at forward): single full sequence (B=1, no padding, no
packing metadata, no cache) — the C5 full-recompute semantics. Sequence packing
/ batched varlen is future work (C8).

``install_zero_diff`` patches forwards at the INSTANCE level (only the given
model is affected; class definitions stay untouched). Install AFTER weight
loading and dtype casting (bf16 bulk + fp32 ``_fp32_params``); re-install if
the modules are rebuilt/re-wrapped.
"""

import os
import types

import torch

from .zero_diff_functions import (
    BILinear,
    FACore,
    GDNCore,
    GemmaAddRMSNormFn,
    GemmaRMSNormFn,
    MoERouted,
    SiluMul,
)

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

# C5 frozen facts for Qwen3.5-35B-A3B (validated bitwise against the vllm
# engine; install asserts the model config matches before patching).
_EXPECTED_FA = dict(
    num_q_heads=16,
    num_kv_heads=2,
    head_dim=256,
    rotary_dim=64,
    mrope_section=[11, 11, 10],
    eps=1e-6,
    kv_block=1088,  # vllm kv cache block size (captured)
    max_seqlen_k=4096,
)
_MAX_T = 4096  # full-recompute window (asserted at forward)
_LM_HEAD_CHUNK = 1024  # vllm compute_prompt_logprobs chunking

_ROPE_CACHE = None  # shared bf16 cos_sin_cache, keyed by (device, params)


def _fa_head_dims(config):
    """FA dims from the text config, asserted against the C5 frozen facts."""
    nq = config.num_attention_heads
    kvh = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // nq
    rotary_dim = getattr(config, "rotary_dim", None)
    if rotary_dim is None:
        prf = getattr(config, "partial_rotary_factor", None)
        rotary_dim = int(head_dim * prf) if prf else head_dim
    mrope_section = list(getattr(config, "mrope_section", None) or [11, 11, 10])
    dims = dict(
        num_q_heads=nq,
        num_kv_heads=kvh,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        mrope_section=mrope_section,
        eps=getattr(config, "rms_norm_eps", 1e-6),
        kv_block=_EXPECTED_FA["kv_block"],
        max_seqlen_k=_EXPECTED_FA["max_seqlen_k"],
    )
    for key, expected in _EXPECTED_FA.items():
        if key in ("kv_block", "max_seqlen_k"):
            continue
        if dims[key] != expected:
            raise ValueError(
                f"0-diff recipe was validated for {key}={expected}, config has {dims[key]}; "
                "refusing to install on an unvalidated configuration."
            )
    return dims


def _get_rope_cache(dev, config):
    """bf16 cos_sin_cache rebuilt via vllm get_rope (bitwise == engine cache)."""
    global _ROPE_CACHE
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope

    rope_theta = getattr(config, "rope_theta", None) or 10000000.0
    partial = getattr(config, "partial_rotary_factor", None) or 0.25
    key = (str(dev), rope_theta, tuple(_fa_head_dims(config)["mrope_section"]), partial)
    if _ROPE_CACHE is not None and _ROPE_CACHE[0] == key:
        return _ROPE_CACHE[1]
    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(
            head_size=_fa_head_dims(config)["head_dim"],
            max_position=262144,
            rope_parameters={
                "mrope_interleaved": True,
                "mrope_section": list(_fa_head_dims(config)["mrope_section"]),
                "rope_type": "default",
                "rope_theta": rope_theta,
                "partial_rotary_factor": partial,
            },
            dtype=torch.bfloat16,
        )
    cache = rope.cos_sin_cache.to(dev)
    _ROPE_CACHE = (key, cache)
    return cache


# ----------------------------------------------------------------------------
# GDN layer (C5 surgery ① + graph): CPAwareGatedDeltaNet instance forward
# ----------------------------------------------------------------------------
def gdn_zero_diff_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    assert cache_params is None, "0-diff recipe is full-recompute only"
    was_3d = hidden_states.ndim == 3
    x = hidden_states.reshape(-1, hidden_states.shape[-1])

    # live weight constants (cat keeps the autograd graph to the parameters;
    # same layout the C5 gate validated bitwise)
    w_qkvz = torch.cat([self.in_proj_qkv.weight, self.in_proj_z.weight], dim=0)
    w_ba = torch.cat([self.in_proj_b.weight, self.in_proj_a.weight], dim=0)
    conv_w = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )
    # vllm keeps dt_bias in bf16 (the checkpoint stores bf16-representable
    # values, so this cast is exact); A_log stays fp32 like vllm
    dt_bias = self.dt_bias
    if dt_bias.dtype != torch.bfloat16:
        dt_bias = dt_bias.to(torch.bfloat16)

    dims = dict(
        num_k_heads=self.num_k_heads,
        num_v_heads=self.num_v_heads,
        head_k_dim=self.head_k_dim,
        head_v_dim=self.head_v_dim,
        conv_kernel=self.conv_kernel_size,
        eps=self.layer_norm_epsilon,
    )
    mixed_qkvz = BILinear.apply(x, w_qkvz)
    ba = BILinear.apply(x, w_ba)
    core = GDNCore.apply(mixed_qkvz, ba, conv_w, self.A_log, dt_bias, self.norm.weight, dims)
    out = BILinear.apply(core, self.out_proj.weight)
    if was_3d:
        out = out.reshape(hidden_states.shape)
    return out


# ----------------------------------------------------------------------------
# MoE block (C5 surgery ③ + graph): component MoE instance forward
# ----------------------------------------------------------------------------
def moe_zero_diff_forward(self, hidden_states, padding_mask=None, cp_mesh=None):
    assert padding_mask is None, "0-diff recipe requires a full (unpadded) sequence"
    was_3d = hidden_states.ndim == 3
    x = hidden_states.reshape(-1, hidden_states.shape[-1])

    # NeMo expert layout is transposed vs the vllm/HF layout: pass views
    # ([E, H, 2N] -> [E, 2N, H]; [E, N, H] -> [E, H, N]) — the Triton kernel
    # and the backward matmuls consume them through strides
    w13 = self.experts.gate_and_up_projs.transpose(1, 2)  # [E, 2N, H] view
    w2 = self.experts.down_projs.transpose(1, 2)  # [E, H, N] view
    E = self.experts.n_routed_experts
    N = w13.shape[1] // 2
    TOPK = self.gate.topk

    logits = BILinear.apply(x, self.gate.weight)
    routed = MoERouted.apply(x, logits, w13, w2, E, N, TOPK)

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


# ----------------------------------------------------------------------------
# FA layer (C5 surgery ④ + graph): _Qwen3_5MoeAttention instance forward
# ----------------------------------------------------------------------------
def fa_zero_diff_forward(self, hidden_states=None, x=None, freqs_cis=None, attention_mask=None, **kwargs):
    hs = hidden_states if hidden_states is not None else x
    assert hs is not None, "hidden_states required"
    was_3d = hs.ndim == 3
    x2 = hs.reshape(-1, hs.shape[-1])
    T = x2.shape[0]
    assert T <= _MAX_T, f"T={T} exceeds the 0-diff recompute window {_MAX_T}"
    dev = x2.device

    head_dims = self._zd_head_dims
    q_size = head_dims["num_q_heads"] * head_dims["head_dim"]
    qg_size = 2 * q_size
    kv_size = head_dims["num_kv_heads"] * head_dims["head_dim"]
    cos_sin = _get_rope_cache(dev, self._zd_text_config)

    # merged qkv GEMM — per-call cat of the live weights (q_proj carries the
    # per-head q|gate interleave, matching the validated C5 layout)
    w_qkv = torch.cat(
        [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0
    )
    qkv = BILinear.apply(x2, w_qkv)
    q_gate, k_in, v = qkv.split([qg_size, kv_size, kv_size], dim=-1)
    gated = FACore.apply(
        q_gate, k_in, v, self.q_norm.weight, self.k_norm.weight, cos_sin, head_dims, T
    )
    output = BILinear.apply(gated, self.o_proj.weight)
    if was_3d:
        output = output.reshape(hs.shape)
    return output


# ----------------------------------------------------------------------------
# Decoder wiring (C5 surgery ② + graph): text backend instance forward
# ----------------------------------------------------------------------------
def text_backend_zero_diff_forward(
    self,
    input_ids=None,
    *,
    inputs_embeds=None,
    attention_mask=None,
    position_ids=None,
    cache_position=None,
    padding_mask=None,
    past_key_values=None,
    use_cache=None,
    **attn_kwargs,
):
    """vllm-replica decoder loop with the autograd graph restored.

    vllm defers every residual add into the FOLLOWING Gemma RMSNorm
    (flashinfer gemma_fused_add_rmsnorm): the kernel adds residual += input
    in fp32 and normalizes the UNROUNDED fp32 sum — replicated here via
    ``GemmaAddRMSNormFn``.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeModelOutputWithPast,
    )

    assert (input_ids is None) != (inputs_embeds is None), "exactly one input"
    assert past_key_values is None and not use_cache, "full-recompute only"
    assert attention_mask is None and padding_mask is None, "single full sequence"
    for key in ("cu_seqlens", "max_seqlen", "_packed_seq_ids", "packed_token_indices"):
        assert key not in attn_kwargs, f"packing metadata '{key}' unsupported by the 0-diff recipe"

    # The trainer engine wraps the whole module call in torch.autocast(bf16);
    # the recipe's ops (torch.sigmoid, +, cat) must run at their native bf16
    # to keep the validated bitwise numerics, so disable autocast locally.
    # (The recipe's kernels are CUDA-only, so a fixed device_type is fine.)
    with torch.autocast(device_type="cuda", enabled=False):
        if inputs_embeds is None:
            hidden = self.embed_tokens(input_ids)  # [B, L, H]
        else:
            hidden = inputs_embeds
        B, L, H = hidden.shape
        assert B == 1, "0-diff recipe is single-sequence"

        cur = hidden.reshape(-1, H)
        residual = None
        for layer in self.layers.values():
            w1 = layer.input_layernorm.weight
            w2 = layer.post_attention_layernorm.weight
            eps = layer.input_layernorm.eps

            if residual is None:
                residual = cur
                y = GemmaRMSNormFn.apply(cur, w1, eps)
            else:
                y, residual = GemmaAddRMSNormFn.apply(cur, residual, w1, eps)

            if layer.layer_type == "linear_attention":
                attn_out = layer.linear_attn(hidden_states=y, attention_mask=None)
            else:
                attn_out = layer.self_attn(hidden_states=y)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            attn_out = attn_out.reshape(-1, H)

            y2, residual = GemmaAddRMSNormFn.apply(attn_out, residual, w2, eps)

            moe_out = layer.mlp(y2.reshape(B, L, H))
            if isinstance(moe_out, tuple):
                moe_out = moe_out[0]
            cur = moe_out.reshape(-1, H)

        y_final, _ = GemmaAddRMSNormFn.apply(cur, residual, self.norm.weight, self.norm.eps)
        return Qwen3_5MoeModelOutputWithPast(
            last_hidden_state=y_final.reshape(B, L, H),
            past_key_values=None,
            rope_deltas=None,
        )


# ----------------------------------------------------------------------------
# lm_head (C5 surgery ⑤ + graph): chunked BI GEMM, vllm-style
# ----------------------------------------------------------------------------
def lm_head_zero_diff_forward(self, hidden_states):
    """vllm lm_head replica: chunked BI GEMM, bf16 logits [*, vocab]."""
    x = hidden_states.reshape(-1, hidden_states.shape[-1])  # [T, H]
    outs = []
    for a in range(0, x.shape[0], _LM_HEAD_CHUNK):
        b = min(a + _LM_HEAD_CHUNK, x.shape[0])
        outs.append(BILinear.apply(x[a:b], self.weight))
    logits = torch.cat(outs, dim=0)
    return logits.reshape(*hidden_states.shape[:-1], -1)


# ----------------------------------------------------------------------------
# installer
# ----------------------------------------------------------------------------
def install_zero_diff(model):
    """Patch a loaded ``Qwen3_5MoeForConditionalGeneration`` to the 0-diff recipe.

    Instance-level patching: only ``model`` is affected. The model must
    already hold bf16 weights with fp32 ``_fp32_params`` (A_log/dt_bias);
    re-install if modules are rebuilt or re-wrapped (e.g. after a parallelizer
    pass in the trainer).

    Returns a dict of patched module counts.
    """
    backend = model.model.language_model
    assert isinstance(backend.__class__, type), "language_model backend missing"
    config = backend.config
    head_dims = _fa_head_dims(config)

    if getattr(model, "mtp", None) is not None:
        raise ValueError(
            "The 0-diff recipe does not cover the MTP heads; build the model with "
            "num_nextn_predict_layers=0."
        )

    backend.forward = types.MethodType(text_backend_zero_diff_forward, backend)

    n_gdn = n_fa = n_moe = 0
    for layer in backend.layers.values():
        if layer.layer_type == "linear_attention":
            gdn = layer.linear_attn
            gdn.forward = types.MethodType(gdn_zero_diff_forward, gdn)
            n_gdn += 1
        else:
            attn = layer.self_attn
            attn._zd_head_dims = head_dims
            attn._zd_text_config = config
            attn.forward = types.MethodType(fa_zero_diff_forward, attn)
            n_fa += 1
        mlp = layer.mlp
        mlp.forward = types.MethodType(moe_zero_diff_forward, mlp)
        n_moe += 1

    lm_head = model.lm_head
    lm_head.forward = types.MethodType(lm_head_zero_diff_forward, lm_head)

    return dict(gdn=n_gdn, fa=n_fa, moe=n_moe, layers=len(backend.layers))
