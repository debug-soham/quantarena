"""GPU memory and on-disk size.

Measured through transformers (not vLLM, which preallocates a fixed fraction of
the GPU and so tells us nothing about the model itself).
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import ExperimentConfig

log = logging.getLogger(__name__)

GiB = 1024**3
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")


def checkpoint_size_bytes(path: str, revision: str | None = None) -> int | None:
    p = Path(path)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.iterdir() if f.suffix in WEIGHT_SUFFIXES)
    # Baseline lives on the Hub; prefer safetensors and don't double count .bin copies.
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(path, revision=revision, files_metadata=True)
        files = [s for s in info.siblings if s.size]
        st = [s for s in files if s.rfilename.endswith(".safetensors")]
        chosen = st or [s for s in files if s.rfilename.endswith(WEIGHT_SUFFIXES)]
        return sum(s.size for s in chosen)
    except Exception as e:  # offline, gated, etc.
        log.warning("could not get checkpoint size for %s: %s", path, e)
        return None


def measure_memory(cfg: ExperimentConfig) -> dict:
    import torch

    from ..loading import load_model

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    model, _ = load_model(cfg.model_path, cfg.model)
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated() - base
    load_peak = torch.cuda.max_memory_allocated() - base

    # One full-length forward pass (no KV cache) to get activation overhead.
    seq_len = cfg.eval.seq_len
    vocab = model.config.vocab_size
    gen = torch.Generator().manual_seed(cfg.seed)
    ids = torch.randint(0, vocab, (1, seq_len), generator=gen).to(model.device)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(ids, use_cache=False)
    torch.cuda.synchronize()
    fwd_peak = torch.cuda.max_memory_allocated() - base

    disk = checkpoint_size_bytes(cfg.model_path, cfg.model.revision)
    n_params = sum(p.numel() for p in model.parameters())

    del model
    torch.cuda.empty_cache()

    return {
        "weights_gib": round(weights / GiB, 3),
        "load_peak_gib": round(load_peak / GiB, 3),
        "forward_peak_gib": round(fwd_peak / GiB, 3),
        "forward_seq_len": seq_len,
        "disk_gib": round(disk / GiB, 3) if disk is not None else None,
        # packed params for quantized layers, so not the "real" count
        "stored_params": n_params,
    }
