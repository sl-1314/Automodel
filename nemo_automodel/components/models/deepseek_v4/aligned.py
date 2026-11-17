"""Deterministic-training aligned forward for DeepSeek-V4 (C5.4 dual-path).

`DeepseekV4ForCausalLMAligned` subclasses :class:`DeepseekV4ForCausalLM` and
overrides ``forward`` with the SGLang deterministic-kernel thread proven
bitwise against the rollout engine in the consistency ledger (c56-c63):
embed + hc_broadcast -> [mhc boundary -> attention stack -> boundary ->
MoE] x n_layers -> final read -> fused_hc_head + RMSNorm -> bf16 lm-head
GEMM -> value-exact fp32 upcast.

Differences from the base class's numeric path (which is left untouched):

* every value is produced by direct SGLang kernel calls chained here, under
  the engine's batch-invariant mode, so the training forward takes the exact
  same operator branches as rollout;
* the parameters are the CONTAINER'S OWN: ``self.state_dict(keep_vars=True)``
  is mapped to the engine file layout by an autograd-transparent inverse
  keymap (see ``_map_native_to_file``); all transforms (cat/split/transpose/
  contiguous/dtype cast) keep the autograd graph alive, so this forward is
  usable as the grad-enabled actor branch;
* discrete scheduling quantities (positions, index tables, page tables,
  compress plans, routing/gate call args) are NOT recomputed: they are
  replayed from the captured ``static`` inputs passed via ``aligned_replay``
  (the C5 contract's fixed "position, mask, routing" from the trajectory);
  they are not gradient targets.

The caller (training loop) is expected to have initialized the SGLang
framework process groups and enabled batch-invariant mode, exactly as an
engine process would (``publish`` + ``bootstrap.init_parallel_runtime`` +
``enable_batch_invariant_mode``).

Output contract: ``DeepseekV4CausalLMOutput`` with ``logits`` of shape
``[T, vocab]`` in float32 — the value-exact upcast of the bf16 GEMM result,
matching the engine's ``_copy_logits_to_buffer`` stage.  The logprob
(log_softmax + gather) is deliberately NOT computed here: it belongs to the
loss side of the training framework, mirroring the native seam where the
model returns logits and the loss module applies ``F.log_softmax(...,
dtype=torch.float32)``.
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F

from nemo_automodel.components.models.deepseek_v4.model import (
    DeepseekV4CausalLMOutput,
    DeepseekV4ForCausalLM,
)

# ---------------------------------------------------------------------------
# Inverse keymap: Automodel native state dict -> SGLang engine file layout.
#
# Autograd-transparent port of the ledger's c51 keymap (native -> file
# direction).  Every transform preserves the graph: cat/split keep grad_fn,
# ``.T.contiguous()`` yields a differentiable clone, dtype casts are
# value-exact (fp32 islands <-> bf16 files hold bf16-representable values).
# Integer tables (tid2eid) are not gradient targets.
# ---------------------------------------------------------------------------

_HL = r"model\.layers\.(\d+)"
_IDX_W = r"(wq_b|weights_proj)"

_INV_RULES = [
    (re.compile(r"^model\.hc_head\.hc_(base|fn|scale)$"),
     lambda m, t: [(f"model.hc_head_{m.group(1)}", t)]),
    (re.compile(rf"^{_HL}\.(attn|ffn)_hc\.(base|fn|scale)$"),
     lambda m, t: [(f"model.layers.{m.group(1)}.hc_{m.group(2)}_{m.group(3)}", t)]),
    (re.compile(rf"^{_HL}\.mlp\.gate\.tid2eid$"),
     lambda m, t: [(f"model.layers.{m.group(1)}.mlp.topk.tid2eid", _to_int32(t))]),
    (re.compile(rf"^{_HL}\.self_attn\.sinks_param\.weight$"),
     lambda m, t: [(f"model.layers.{m.group(1)}.self_attn.attn_sink", t)]),
    (re.compile(rf"^{_HL}\.self_attn\.compressor\.ape_param\.weight$"),
     lambda m, t: [(f"model.layers.{m.group(1)}.self_attn.compressor.ape", t)]),
    (re.compile(rf"^{_HL}\.self_attn\.compressor\.kv_norm\.weight$"),
     lambda m, t: [(f"model.layers.{m.group(1)}.self_attn.compressor.norm.weight", t)]),
    (re.compile(rf"^{_HL}\.self_attn\.compressor\.indexer\.ape_param\.weight$"),
     lambda m, t: [(f"model.layers.{m.group(1)}.self_attn.indexer.compressor.ape", t)]),
    (re.compile(rf"^{_HL}\.self_attn\.compressor\.indexer\.kv_norm\.weight$"),
     lambda m, t: [(f"model.layers.{m.group(1)}.self_attn.indexer.compressor.norm.weight", t)]),
    (re.compile(rf"^{_HL}\.self_attn\.compressor\.indexer\.({_IDX_W})\.weight$"),
     lambda m, t: [(f"model.layers.{m.group(1)}.self_attn.indexer.{m.group(2)}.weight", t)]),
]

_INV_WKV_PAIR = re.compile(
    rf"^{_HL}\.self_attn\.compressor\.(indexer\.)?(wkv|wgate)\.weight$")

_GU_STACK = re.compile(rf"^{_HL}\.mlp\.experts\.gate_and_up_projs$")
_DOWN_STACK = re.compile(rf"^{_HL}\.mlp\.experts\.down_projs$")

# The engine's dsv4 loader permutes ratio-4 compressor ape elements in place
# when it consumes the file tensor; the file/native values are PRE-permutation,
# so the aligned thread applies the same hotfix at the consumption site.
_ASSERT_KEYS = ("hidden_size", "num_hidden_layers", "hc_mult", "rms_norm_eps",
                "hc_eps", "hc_sinkhorn_iters", "q_lora_rank", "head_dim",
                "num_attention_heads", "o_groups", "o_lora_rank",
                "qk_rope_head_dim", "vocab_size")


def _to_int32(t: torch.Tensor) -> torch.Tensor:
    t32 = t.to(torch.int32)
    assert t32.to(torch.int64).equal(t.to(torch.int64)), "tid2eid not int32-lossless"
    return t32


def _map_native_to_file(sd: dict) -> dict:
    """Map a full Automodel-native state dict to the engine file layout.

    Autograd-transparent: input tensors are never detached; transformed
    outputs carry ``grad_fn`` back to the container's parameters.  Raises on
    unmapped leftovers.
    """
    out: dict = {}
    fuse: dict[str, dict[str, torch.Tensor]] = {}
    experts: dict[int, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        m = _INV_WKV_PAIR.match(k)
        if m:
            prefix = f"model.layers.{m.group(1)}.self_attn." + (
                "indexer.compressor" if m.group(2) else "compressor")
            fuse.setdefault(prefix, {})[m.group(3)] = v
            continue
        if _GU_STACK.match(k) or _DOWN_STACK.match(k):
            layer = int(k.split(".")[2])
            experts.setdefault(layer, {})[k.split(".")[-1]] = v
            continue
        for pat, handler in _INV_RULES:
            m = pat.match(k)
            if m:
                for nk, nt in handler(m, v):
                    assert nk not in out, f"duplicate source key {nk}"
                    out[nk] = nt.contiguous() if isinstance(nt, torch.Tensor) else nt
                break
        else:
            assert k not in out, f"duplicate source key {k}"
            out[k] = v
    for prefix, parts in fuse.items():
        assert set(parts) == {"wkv", "wgate"}, f"incomplete wkv/gate at {prefix}"
        out[f"{prefix}.wkv_gate.weight"] = torch.cat(
            [parts["wkv"], parts["wgate"]], dim=0)
    for layer, d in sorted(experts.items()):
        assert set(d) == {"gate_and_up_projs", "down_projs"}, (
            f"layer {layer} experts: incomplete {sorted(d)}")
        gu, down = d["gate_and_up_projs"], d["down_projs"]
        inter = gu.shape[-1] // 2
        assert gu.shape[-1] == 2 * inter and gu.shape[0] == down.shape[0]
        for e in range(gu.shape[0]):
            lp = f"model.layers.{layer}.mlp.experts.{e}"
            out[f"{lp}.gate_proj.weight"] = gu[e][:, :inter].T.contiguous()
            out[f"{lp}.up_proj.weight"] = gu[e][:, inter:].T.contiguous()
            out[f"{lp}.down_proj.weight"] = down[e].T.contiguous()
    return out


def _ape_hotfix(ape: torch.Tensor) -> torch.Tensor:
    """Engine loader's in-place ratio-4 ape element permutation."""
    if ape.shape[0] != 4:
        return ape
    a = torch.chunk(ape, 2, dim=-1)
    a = torch.cat([a[0], a[1]], dim=0)
    return a.view(ape.shape[0], -1)


def _td(name: str):
    return getattr(torch, name.replace("torch.", ""))


def _cuda_back(t):
    if isinstance(t, torch.Tensor):
        return t.cuda() if t.device.type == "cpu" else t
    if isinstance(t, (list, tuple)):
        return type(t)(_cuda_back(x) for x in t)
    return t


def _plan_to_cuda(plan):
    d = plan._asdict()
    return type(plan)(**{k: _cuda_back(v) for k, v in d.items()})


class DeepseekV4ForCausalLMAligned(DeepseekV4ForCausalLM):
    """DSV4 container whose forward runs the deterministic aligned thread.

    Registered under the ``DeepseekV4ForCausalLM`` architecture name by the
    training driver (``get_registry().register(..., exist_ok=True)``) so the
    native framework entry (``NeMoAutoModelForCausalLM.from_pretrained``)
    constructs this class with the standard config + state-dict-adapter load
    path unchanged.  The base class's numeric path is untouched; only
    ``forward`` is overridden.
    """

    # callsite-hit instrumentation for the dual-path receipt: the driver
    # asserts this counter advanced, proving the aligned forward (not the
    # base numeric path) actually executed.
    aligned_forward_calls: int = 0
    # theta-sync receipt: number of file-layout keys verified bitwise on the
    # most recent forward that was given reference weights.
    aligned_last_sync: dict | None = None

    def aligned_file_weights(self) -> dict:
        """Container parameters mapped to the engine file layout (grad-safe)."""
        return _map_native_to_file(self.state_dict(keep_vars=True))

    def forward(
        self,
        input_ids=None,
        *mtp_embed_inputs,
        position_ids=None,
        attention_mask=None,
        padding_mask=None,
        logits_to_keep=0,
        output_hidden_states=None,
        aligned_replay: dict | None = None,
        aligned_ref_weights: dict | None = None,
        **attn_kwargs,
    ):
        """Aligned deterministic forward.

        Args:
            input_ids: the packed token histories — either a list of per-
                request 1-D id sequences (THD packing; their concatenation is
                the flat batch) or a flat 1-D id tensor plus
                ``aligned_seq_lens`` in ``aligned_replay``.
            aligned_replay: the captured discrete replay inputs (the c63
                ``static.pt`` dict: positions, index tables, page tables,
                compress plans, kernel configs, routing/gate call args).
                Required — this forward replays, never recomputes, the
                scheduling quantities.
            aligned_ref_weights: optional file-layout reference weights; when
                given, every mapped tensor is cast to the reference dtype and
                asserted bitwise-equal (the theta-sync receipt) before use.

        Returns:
            ``DeepseekV4CausalLMOutput`` with float32 ``logits`` [T, vocab]
            (value-exact upcast of the bf16 GEMM result).
        """
        if aligned_replay is None:
            raise ValueError(
                "DeepseekV4ForCausalLMAligned.forward requires aligned_replay "
                "(captured discrete scheduling inputs); the base numeric path "
                "is the one to use without it")
        DeepseekV4ForCausalLMAligned.aligned_forward_calls += 1

        if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1:
            seq_lens = aligned_replay.get("aligned_seq_lens")
            if seq_lens is None:
                raise ValueError("flat input_ids require aligned_seq_lens")
            histories = [input_ids[o:o + n].tolist()
                         for o, n in zip(_offsets(seq_lens), seq_lens)]
        else:
            histories = [list(h) for h in input_ids]
        static = aligned_replay

        # config consistency: the replayed static inputs were captured under
        # the engine's config; the container must agree on every scalar the
        # thread dereferences.
        cfg = static["config"]
        for k in _ASSERT_KEYS:
            if k in cfg:
                assert getattr(self.config, k) == cfg[k], (
                    f"config mismatch on {k}: container "
                    f"{getattr(self.config, k)} != replay {cfg[k]}")

        fw = self._aligned_fw(static, aligned_ref_weights)
        logits = self._aligned_thread(histories, static, fw)
        return DeepseekV4CausalLMOutput(logits=logits)

    # -- weight plumbing ----------------------------------------------------

    def _aligned_fw(self, static, ref_weights):
        """fw(key) -> file-layout, file-dtype, CUDA, grad-connected tensor."""
        mapped = self.aligned_file_weights()
        sync = {"n_keys": len(mapped), "n_ref": 0, "n_cast": 0}
        if ref_weights is not None:
            missing = set(mapped) - set(ref_weights)
            assert not missing, f"mapped keys missing from reference: {sorted(missing)[:5]}"
            for k, ref in ref_weights.items():
                assert k in mapped, f"reference key not produced by container: {k}"
                t = mapped[k]
                if t.dtype != ref.dtype:
                    t = t.to(ref.dtype)  # value-exact fp32<->bf16 island cast
                    sync["n_cast"] += 1
                if t.device != ref.device:
                    ref = ref.to(t.device)
                assert torch.equal(t, ref), (
                    f"theta-sync FAILED at {k}: container value != file weight")
                mapped[k] = t
                sync["n_ref"] += 1
        DeepseekV4ForCausalLMAligned.aligned_last_sync = sync

        def fw(key):
            t = mapped[key]
            return t.cuda() if t.device.type == "cpu" else t

        return fw

    # -- the deterministic thread (ledger c63, engine objects -> replay) ----

    def _aligned_thread(self, histories, static, fw):
        import deep_gemm  # noqa: F401
        from sglang.kernels.ops.attention import flash_mla_interface
        from sglang.kernels.ops.attention.dsv4 import (
            compress_forward, fused_q_indexer_rope_hadamard_quant,
            topk_transform_paged,
        )
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
        from sglang.kernels.ops.attention.dsv4.norm_rope import (
            fused_k_norm_rope_flashmla, fused_q_norm_rope, fused_rope_inplace,
        )
        from sglang.srt.layers.attention.dsv4 import compressor_v2 as cv2
        from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32
        from sglang.kernels.ops.activation.silu_and_mul_clamp import (
            silu_and_mul_clamp,
        )
        from sglang.kernels.ops.layernorm.mhc import hc_broadcast, mhc_post, mhc_pre
        from sglang.kernels.ops.layernorm.mhc_head import fused_hc_head
        from sglang.kernels.ops.moe.dsv4 import hash_topk
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
        from sglang.srt.batch_invariant_ops import (
            is_batch_invariant_mode_enabled,
            rms_norm_batch_invariant,
        )
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _create_flashmla_metadata, _fill_deterministic_flashmla_sched,
        )
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
            apply_mhc_post_pre_boundary,
        )
        from sglang.srt.models.deepseek_v4_mhc import MHC_POST_MULT_VALUE

        assert is_batch_invariant_mode_enabled(), (
            "the aligned forward requires the engine's batch-invariant torch-op "
            "overrides (enable_batch_invariant_mode) — without them the GEMMs "
            "and log_softmax take different reduction branches than rollout")

        cfg = static["config"]
        n_layers = cfg["num_hidden_layers"]
        flat_ids = torch.tensor([t for h in histories for t in h],
                                dtype=torch.int64)
        T = flat_ids.shape[0]

        # ---- replayed static inputs ---------------------------------------
        positions = static["positions"].cuda()
        swa_loc = static["swa_loc"].cuda().to(static["swa_loc_dtype"])
        swa_idx, swa_len = static["swa_idx"].cuda(), static["swa_len"].cuda()
        c4_out_loc = static["c4_out_loc"].cuda()
        c128_out_loc = static["c128_out_loc"].cuda()
        c128_idx, c128_len = static["c128_idx"].cuda(), static["c128_len"].cuda()
        c4sl1d = static["c4sl1d"].cuda()
        c4sl = c4sl1d.unsqueeze(-1)
        page_table = static["page_table"].cuda()
        dg_meta = _cuda_back(static["dg_meta"])
        plan_c4 = _plan_to_cuda(static["plan_c4"])
        plan_c128 = _plan_to_cuda(static["plan_c128"])
        c4_sparse_lens = static["c4_sparse_topk_lengths"].cuda()

        swa_page, swa_bpt = static["swa_page"], static["swa_bpt"]
        swa_row, e_row, i_row, e3_row = (static["swa_row"], static["e_row"],
                                         static["i_row"], static["e3_row"])
        kv_layout = KVLayout[static["kv_layout"].split(".")[-1]]
        radix_dtype = _td(static["radix_dtype"])
        extra_dtype = {int(k): _td(v) for k, v in static["extra_dtype"].items()}
        index_dtype = _td(static["index_dtype"])
        ep = {int(k): v for k, v in static["ep"].items()}
        ebpt = {int(k): v for k, v in static["ebpt"].items()}
        head_dim_v = static["head_dim_v"]
        softmax_scale = static["softmax_scale"]
        swa_interleaved = static["swa_interleaved"]
        layer_ratio = {int(k): v for k, v in static["layer_ratio"].items()}
        is_hash = {int(k): v for k, v in static["is_hash"].items()}
        mrc = {int(k): v for k, v in static["moe_runner_config"].items()}
        hash_calls, gate_calls = static["hash_calls"], static["gate_calls"]
        lm_head_dtype = _td(static["lm_head_dtype"])
        iln_dtype = _td(static["input_layernorm_dtype"])

        # right-sized glue pools: the paged kernels only dereference the pages
        # the index tables point at, so zero-filled small buffers reproduce
        # the engine pools bitwise at every dereferenced location.
        swa_rows = int(max(swa_idx.max().item(),
                           swa_loc.max().item() // swa_page) + 1)
        c4_rows = int(max(page_table.max().item(),
                          c4_out_loc.max().item() // 64) + 1)
        c128_rows = int(max(c128_idx.max().item(),
                            c128_out_loc.max().item() // 2) + 1)

        # ---- 0) token ids -> hs_init --------------------------------------
        emb = F.embedding(flat_ids.cuda(), fw("model.embed_tokens.weight"))
        hs_init = hc_broadcast(emb, cfg["hc_mult"])

        state = {"streams": hs_init, "post": None, "comb": None}
        hash_j, gate_j = 0, 0

        for i in range(n_layers):
            ratio = layer_ratio[i]

            # ---- 1) attention boundary -> x_attn --------------------------
            if i == 0:
                # mhc_pre reads its norm weight as a plain argument; the
                # engine passes the input_layernorm weight held at the
                # engine's storage dtype (bf16 values upcast in the fp32
                # module slot) — replicate values + dtype exactly.
                w_iln = fw("model.layers.0.input_layernorm.weight") \
                    .to(iln_dtype).to(torch.float32)
                post0, comb0, y = mhc_pre(
                    residual=state["streams"],
                    fn=fw("model.layers.0.hc_attn_fn"),
                    hc_scale=fw("model.layers.0.hc_attn_scale"),
                    hc_base=fw("model.layers.0.hc_attn_base"),
                    rms_eps=cfg["rms_norm_eps"],
                    hc_pre_eps=cfg["hc_eps"],
                    hc_sinkhorn_eps=cfg["hc_eps"],
                    hc_post_mult_value=MHC_POST_MULT_VALUE,
                    sinkhorn_repeat=cfg["hc_sinkhorn_iters"],
                    norm_weight=w_iln,
                    norm_eps=cfg["rms_norm_eps"],
                )
                x_attn = y
                state["post"], state["comb"] = post0.squeeze(-1), comb0
            else:
                fused = self._glue_fused(
                    state["value"], state["streams"], state["post"],
                    state["comb"], i, "attn", fw, cfg,
                    apply_mhc_post_pre_boundary, MHC_POST_MULT_VALUE)
                g_streams, x_attn, g_post, g_comb, _ = fused
                state.update(streams=g_streams, post=g_post, comb=g_comb)

            # ---- 2) attention stack ---------------------------------------
            swa_glue = torch.zeros(swa_rows, swa_row, dtype=torch.uint8,
                                   device="cuda")
            # 2a) Q-side
            wq_a = fw(f"model.layers.{i}.self_attn.wq_a.weight")
            wkv = fw(f"model.layers.{i}.self_attn.wkv.weight")
            qkv_a = F.linear(x_attn, torch.cat([wq_a, wkv], dim=0))
            # q_norm: engine holds the weight at its own dtype (bf16 values
            # in the fp32 module slot); the batch-invariant rms_norm kernel
            # is called directly with those values so the weight stays
            # grad-connected (the module copy_ would detach it).
            w_qn = fw(f"model.layers.{i}.self_attn.q_norm.weight") \
                .to(_td(static["norm_dtypes"][i]["q_norm"])) \
                .to(torch.float32)
            q_lora = rms_norm_batch_invariant(
                qkv_a[..., : cfg["q_lora_rank"]], w_qn, cfg["rms_norm_eps"])
            q = F.linear(q_lora, fw(f"model.layers.{i}.self_attn.wq_b.weight")) \
                .view(-1, cfg["num_attention_heads"], cfg["head_dim"])
            q_out = torch.empty_like(q)
            fused_q_norm_rope(q, q_out, cfg["rms_norm_eps"],
                              static["freqs_attn"][i].cuda(), positions)

            # 2b) KV fp8 store into the glue SWA pool
            fused_k_norm_rope_flashmla(
                kv=qkv_a[..., cfg["q_lora_rank"]:],
                kv_weight=fw(f"model.layers.{i}.self_attn.kv_norm.weight")
                .to(_td(static["norm_dtypes"][i]["kv_norm"])),
                eps=cfg["rms_norm_eps"],
                freqs_cis=static["freqs_attn"][i].cuda(),
                positions=positions,
                out_loc=swa_loc,
                kvcache=swa_glue,
                page_size=swa_page,
                layout=kv_layout,
            )

            # 2c) compressors -> glue extra/index pools
            c4_glue_idx = None
            if ratio != 0:
                plan = plan_c128 if ratio == 128 else plan_c4
                out_loc = c128_out_loc if ratio == 128 else c4_out_loc
                specs = []
                if ratio == 4:
                    index_glue = torch.zeros(c4_rows, i_row, dtype=torch.uint8,
                                             device="cuda")
                    extra_glue = torch.zeros(c4_rows, e_row, dtype=torch.uint8,
                                             device="cuda")
                    specs.append(("idx", "self_attn.indexer.compressor",
                                  index_glue, 64))
                    specs.append(("attn", "self_attn.compressor", extra_glue, 64))
                elif ratio == 128:
                    extra3_glue = torch.zeros(c128_rows, e3_row,
                                              dtype=torch.uint8, device="cuda")
                    specs.append(("attn", "self_attn.compressor", extra3_glue, 2))
                for which, pfx, kv_cache, page_size in specs:
                    key = f"{i}:{which}"
                    cp = static["comp_params"][key]
                    w_wkv = fw(f"model.layers.{i}.{pfx}.wkv_gate.weight")
                    w_norm = fw(f"model.layers.{i}.{pfx}.norm.weight")
                    w_ape = _ape_hotfix(fw(f"model.layers.{i}.{pfx}.ape"))
                    kv_score_input = linear_bf16_fp32(x_attn, w_wkv)
                    ksm = static["kv_score_meta"][key]
                    zstate = torch.zeros(ksm["shape"], dtype=_td(ksm["dtype"]),
                                         device="cuda")
                    last_dim = 2 * (1 + cp["overlap"]) * cp["head_dim"]
                    kv_compressed = compress_forward(
                        kv_score_buffer=zstate.view(-1, cp["ratio"], last_dim),
                        kv_score_input=kv_score_input,
                        ape=w_ape.view(-1, cp["head_dim"]),
                        plan=plan,
                        head_dim=cp["head_dim"],
                        compress_ratio=cp["ratio"],
                    )
                    cv2.compress_norm_rope_store(
                        kv_compressed, plan,
                        norm_weight=w_norm,
                        norm_eps=cp["norm_eps"],
                        freq_cis=static["freqs_comp"][key].cuda(),
                        out_loc=out_loc,
                        kvcache=kv_cache,
                        page_size=page_size,
                        layout=kv_layout,
                    )

            # 2d) indexer (L2/L4)
            if ratio == 4:
                ip = static["indexer_params"][i]
                w_wq = fw(f"model.layers.{i}.self_attn.indexer.wq_b.weight")
                w_wp = fw(f"model.layers.{i}.self_attn.indexer.weights_proj.weight")
                weights = F.linear(x_attn, w_wp)
                q_idx = F.linear(q_lora, w_wq).view(
                    -1, ip["n_local_heads"], ip["head_dim"])
                weight_scale = (ip["head_dim"] ** -0.5
                                * ip["n_heads"] ** -0.5)
                q_fp8, w_out = fused_q_indexer_rope_hadamard_quant(
                    q_idx, weights, weight_scale,
                    static["freqs_indexer"][i].cuda(), positions)
                kv_view = index_glue.view(index_dtype).view(-1, 64, 1, 132)
                logits = deep_gemm.fp8_paged_mqa_logits(
                    q_fp8.unsqueeze(1), kv_view, w_out.squeeze(2), c4sl,
                    page_table, dg_meta, static["max_cs_len"], False)
                out_idx = torch.full((T, 512), -1, dtype=torch.int32,
                                     device="cuda")
                out_raw = torch.empty_like(out_idx)
                topk_transform_paged(logits, c4sl1d, page_table, out_idx,
                                     static["comp_page_size"], out_raw)
                c4_glue_idx = out_idx

            # 2e) flash_mla on the glue pools
            if ratio != 0:
                if ratio == 128:
                    extra_k = extra3_glue[:, : ep[3] * ebpt[3]].view(
                        extra3_glue.shape[0], ep[3], 1, ebpt[3]).view(
                        extra_dtype[3])
                    extra_idx = c128_idx
                    extra_len = c128_len
                else:
                    extra_k = extra_glue[:, : ep[2] * ebpt[2]].view(
                        extra_glue.shape[0], ep[2], 1, ebpt[2]).view(
                        extra_dtype[2])
                    extra_idx = c4_glue_idx
                    extra_len = c4_sparse_lens
            else:
                extra_k = extra_idx = extra_len = None

            indices = swa_idx
            if indices.ndim == 2:
                indices = indices.unsqueeze(1)
            topk_length = swa_len
            if extra_idx is not None and extra_idx.ndim == 2:
                extra_idx = extra_idx.unsqueeze(1)

            meta = _create_flashmla_metadata()
            _fill_deterministic_flashmla_sched(
                meta, q=q_out.unsqueeze(1), indices=indices,
                topk_length=topk_length, extra_indices=extra_idx,
                extra_topk_length=extra_len)
            swa_k = swa_glue.view(radix_dtype)[:, : swa_page * swa_bpt].view(
                swa_glue.shape[0], swa_page, 1, swa_bpt)
            o = flash_mla_interface.flash_mla_with_kvcache(
                q=q_out.unsqueeze(1),
                k_cache=swa_k,
                head_dim_v=head_dim_v,
                block_table=None,
                cache_seqlens=None,
                tile_scheduler_metadata=meta,
                softmax_scale=softmax_scale,
                is_fp8_kvcache=True,
                indices=indices,
                topk_length=topk_length,
                attn_sink=fw(f"model.layers.{i}.self_attn.attn_sink").float(),
                extra_k_cache=extra_k,
                extra_indices_in_kvcache=extra_idx,
                extra_topk_length=extra_len,
                use_legacy=not swa_interleaved,
            )[0].squeeze(1)

            # 2f) O-side
            fused_rope_inplace(o[..., -cfg["qk_rope_head_dim"]:], None,
                               static["freqs_attn"][i].cuda(), positions,
                               inverse=True)
            wo = o.view(o.shape[0], cfg["o_groups"], -1)
            wo_a = fw(f"model.layers.{i}.self_attn.wo_a.weight")
            y = torch.einsum(
                "tgd,grd->tgr", wo,
                wo_a.view(cfg["o_groups"], cfg["o_lora_rank"], -1))
            attn_out = F.linear(y.flatten(1),
                                fw(f"model.layers.{i}.self_attn.wo_b.weight"))

            # ---- 3) ffn boundary -> x_mlp ----------------------------------
            fused = self._glue_fused(
                attn_out, state["streams"], state["post"], state["comb"],
                i, "ffn", fw, cfg, apply_mhc_post_pre_boundary,
                MHC_POST_MULT_VALUE)
            g_streams, x_mlp, g_post, g_comb, _ = fused
            state.update(streams=g_streams, post=g_post, comb=g_comb)

            # ---- 4) MoE -> merged ------------------------------------------
            sh_g = fw(f"model.layers.{i}.mlp.shared_experts.gate_proj.weight")
            sh_u = fw(f"model.layers.{i}.mlp.shared_experts.up_proj.weight")
            sh_d = fw(f"model.layers.{i}.mlp.shared_experts.down_proj.weight")
            gate_up = F.linear(x_mlp, torch.cat([sh_g, sh_u], dim=0))
            act = gate_up.new_empty((x_mlp.shape[0], gate_up.shape[1] // 2))
            silu_and_mul_clamp(gate_up, act, 10.0)
            shared_glue = F.linear(act, sh_d)

            gate_w = fw(f"model.layers.{i}.mlp.gate.weight")
            logits_glue = linear_bf16_fp32(x_mlp, gate_w)
            if is_hash[i]:
                hc = hash_calls[hash_j]
                tid2eid = hc["tid2eid"].cuda()
                w_g, ids_g = hash_topk(logits_glue, hc["input_ids"].cuda(),
                                       tid2eid, *hc["a"], **hc["kw"])
                hash_j += 1
            else:
                gc = gate_calls[gate_j]
                bias_g = fw(
                    f"model.layers.{i}.mlp.gate.e_score_correction_bias"
                ).float()
                w_g, ids_g = moe_fused_gate(logits_glue, bias_g, gc["topk"],
                                            *gc["a"], **gc["kw"])
                gate_j += 1

            gs = torch.stack([fw(f"model.layers.{i}.mlp.experts.{e}.gate_proj"
                                 ".weight") for e in range(8)])
            us = torch.stack([fw(f"model.layers.{i}.mlp.experts.{e}.up_proj"
                                 ".weight") for e in range(8)])
            ds = torch.stack([fw(f"model.layers.{i}.mlp.experts.{e}.down_proj"
                                 ".weight") for e in range(8)])
            w13 = torch.cat([gs, us], dim=1)
            topk_output = StandardTopKOutput(w_g, ids_g, logits_glue)
            # fused_experts is in-place on hidden_states — feed a clone
            routed_glue = fused_experts(x_mlp.clone(), w13, ds,
                                        topk_output, mrc[i])
            merged = routed_glue.clone()
            merged += shared_glue
            state["value"] = merged

            del swa_glue
            if ratio == 4:
                del index_glue, extra_glue
            if ratio == 128:
                del extra3_glue
            torch.cuda.empty_cache()

        # ---- 5) final read + tail -----------------------------------------
        R = mhc_post(state["value"], state["streams"], state["post"],
                     state["comb"])
        y = fused_hc_head(
            R.view(R.shape[0], cfg["hc_mult"], cfg["hidden_size"]).contiguous(),
            fw("model.hc_head_fn"),
            fw("model.hc_head_scale"),
            fw("model.hc_head_base"),
            norm_eps=cfg["rms_norm_eps"],
            hc_eps=cfg["hc_eps"],
        )
        # final RMSNorm: the engine consumes the weight in the fp32 module
        # slot (file fp32 values); the batch-invariant kernel is called
        # directly so the weight stays grad-connected.
        states = rms_norm_batch_invariant(
            y, fw("model.norm.weight").to(torch.float32),
            cfg["rms_norm_eps"])
        # lm-head: bf16 GEMM (bf16 rounding) -> value-exact fp32 upcast, the
        # engine's logits-buffer stage.  log_softmax + gather stay on the
        # loss side of the training framework.
        W = fw("lm_head.weight")
        if W.dtype != lm_head_dtype:
            W = W.to(lm_head_dtype)
        logits = torch.matmul(states.to(W.dtype), W.T)
        return logits.to(torch.float32)

    @staticmethod
    def _glue_fused(value, streams, post, comb, i, kind, fw, cfg,
                    apply_mhc_post_pre_boundary, MHC_POST_MULT_VALUE):
        mod = "input_layernorm" if kind == "attn" \
            else "post_attention_layernorm"
        return apply_mhc_post_pre_boundary(
            value, streams, post, comb,
            fw(f"model.layers.{i}.hc_{kind}_fn"),
            fw(f"model.layers.{i}.hc_{kind}_scale"),
            fw(f"model.layers.{i}.hc_{kind}_base"),
            cfg["hc_mult"], cfg["rms_norm_eps"], cfg["hc_eps"],
            MHC_POST_MULT_VALUE, cfg["hc_sinkhorn_iters"],
            fw(f"model.layers.{i}.{mod}.weight").to(torch.bfloat16),
            cfg["rms_norm_eps"],
            fn_transpose=(kind == "ffn"),
        )


def _offsets(seq_lens):
    off, acc = [], 0
    for n in seq_lens:
        off.append(acc)
        acc += n
    return off
