"""Quantization backends.

Each backend is a function ``(cfg, out_dir) -> None`` that writes a checkpoint
loadable by transformers and vLLM. Backends are imported lazily so that only
the library for the method being run has to be installed.
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from pathlib import Path

from ..config import ExperimentConfig

log = logging.getLogger(__name__)

BACKENDS = {
    "awq": "quantarena.quantize.awq:quantize_awq",
}

MARKER = "quantarena.json"


def _resolve(method: str):
    try:
        target = BACKENDS[method]
    except KeyError:
        raise ValueError(f"no quantization backend for method {method!r}") from None
    module, _, fn = target.partition(":")
    return getattr(importlib.import_module(module), fn)


def is_quantized(cfg: ExperimentConfig) -> bool:
    ckpt = cfg.checkpoint_path
    if ckpt is None:
        return True
    marker = ckpt / MARKER
    if not marker.exists():
        return False
    meta = json.loads(marker.read_text())
    return meta.get("quant_fingerprint") == cfg.quant_fingerprint()


def quantize(cfg: ExperimentConfig, force: bool = False) -> Path | None:
    out = cfg.checkpoint_path
    if out is None:
        log.info("method is 'none', nothing to quantize")
        return None
    if is_quantized(cfg) and not force:
        log.info("checkpoint %s is up to date, skipping", out)
        return out

    fn = _resolve(cfg.quant.method)
    out.mkdir(parents=True, exist_ok=True)
    log.info("quantizing %s with %s -> %s", cfg.model.hf_id, cfg.quant.label, out)
    t0 = time.perf_counter()
    fn(cfg, out)
    elapsed = time.perf_counter() - t0

    (out / MARKER).write_text(
        json.dumps(
            {
                "run_id": cfg.run_id,
                "quant_fingerprint": cfg.quant_fingerprint(),
                "quantize_time_s": round(elapsed, 1),
                "quant": cfg.to_dict()["quant"],
                "calib": cfg.to_dict()["calib"],
            },
            indent=2,
        )
    )
    log.info("done in %.1f min", elapsed / 60)
    return out
