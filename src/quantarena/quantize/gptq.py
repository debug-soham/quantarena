from __future__ import annotations

from pathlib import Path

from ..config import ExperimentConfig
from ..data import calibration_texts


def quantize_gptq(cfg: ExperimentConfig, out: Path) -> None:
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    m = cfg.model
    opts = cfg.quant.options
    extra = {"revision": m.revision} if m.revision else {}

    tok = AutoTokenizer.from_pretrained(m.hf_id, trust_remote_code=m.trust_remote_code, **extra)

    qcfg = QuantizeConfig(
        bits=cfg.quant.bits,
        group_size=cfg.quant.group_size,
        desc_act=opts.get("desc_act", True),
        sym=opts.get("sym", True),
        damp_percent=opts.get("damp_percent", 0.01),
    )

    # Tokenize ourselves so the calibration length matches AWQ exactly.
    calib = []
    for text in calibration_texts(cfg.calib, cfg.seed):
        enc = tok(text, truncation=True, max_length=cfg.calib.seq_len)
        calib.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})

    model = GPTQModel.load(m.hf_id, qcfg, trust_remote_code=m.trust_remote_code, **extra)
    model.quantize(calib, batch_size=opts.get("batch_size", 1))
    model.save(str(out))
    tok.save_pretrained(str(out))
