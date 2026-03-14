from __future__ import annotations

from pathlib import Path

from ..config import ExperimentConfig
from ..data import calibration_texts


def quantize_awq(cfg: ExperimentConfig, out: Path) -> None:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    m = cfg.model
    opts = cfg.quant.options
    extra = {"revision": m.revision} if m.revision else {}

    tok = AutoTokenizer.from_pretrained(m.hf_id, trust_remote_code=m.trust_remote_code, **extra)
    model = AutoAWQForCausalLM.from_pretrained(
        m.hf_id,
        trust_remote_code=m.trust_remote_code,
        low_cpu_mem_usage=True,
        use_cache=False,
        **extra,
    )

    quant_config = {
        "zero_point": opts.get("zero_point", True),
        "q_group_size": cfg.quant.group_size,
        "w_bit": cfg.quant.bits,
        "version": opts.get("version", "GEMM"),
    }
    texts = calibration_texts(cfg.calib, cfg.seed)

    model.quantize(
        tok,
        quant_config=quant_config,
        calib_data=texts,
        max_calib_samples=cfg.calib.n_samples,
        max_calib_seq_len=cfg.calib.seq_len,
    )
    model.save_quantized(str(out))
    tok.save_pretrained(str(out))
