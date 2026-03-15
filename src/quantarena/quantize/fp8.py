from __future__ import annotations

from pathlib import Path

from ..config import ExperimentConfig
from ..data import calibration_texts


def _oneshot():
    try:
        from llmcompressor import oneshot
    except ImportError:  # older llmcompressor releases
        from llmcompressor.transformers import oneshot
    return oneshot


def quantize_fp8(cfg: ExperimentConfig, out: Path) -> None:
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = cfg.model
    opts = cfg.quant.options
    extra = {"revision": m.revision} if m.revision else {}
    scheme = opts.get("scheme", "FP8_DYNAMIC")

    tok = AutoTokenizer.from_pretrained(m.hf_id, trust_remote_code=m.trust_remote_code, **extra)
    model = AutoModelForCausalLM.from_pretrained(
        m.hf_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=m.trust_remote_code,
        **extra,
    )

    recipe = QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=opts.get("ignore", ["lm_head"]),
    )
    oneshot = _oneshot()

    if scheme == "FP8_DYNAMIC":
        # activation scales are computed at runtime, weights are per-channel: no data needed
        oneshot(model=model, recipe=recipe)
    else:
        from datasets import Dataset

        texts = calibration_texts(cfg.calib, cfg.seed)
        ds = Dataset.from_dict({"text": texts}).map(
            lambda b: tok(b["text"], truncation=True, max_length=cfg.calib.seq_len),
            batched=True,
            remove_columns=["text"],
        )
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=cfg.calib.seq_len,
            num_calibration_samples=len(texts),
        )

    model.save_pretrained(str(out), save_compressed=True)
    tok.save_pretrained(str(out))
