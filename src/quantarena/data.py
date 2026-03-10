from __future__ import annotations

import logging
import random

from .config import CalibSpec, EvalSpec

log = logging.getLogger(__name__)


def load_texts(dataset: str, subset: str | None, split: str, text_field: str) -> list[str]:
    from datasets import load_dataset

    if subset:
        ds = load_dataset(dataset, subset, split=split)
    else:
        ds = load_dataset(dataset, split=split)
    return list(ds[text_field])


def eval_corpus(spec: EvalSpec) -> str:
    # Same join as the HF perplexity guide so numbers are comparable with other reports.
    return "\n\n".join(load_texts(spec.dataset, spec.subset, spec.split, spec.text_field))


def calibration_texts(spec: CalibSpec, seed: int, min_words: int = 64) -> list[str]:
    """Sample ``n_samples`` documents for calibration, deterministically for a given seed.

    Very short documents are skipped since they mostly produce padding. All three
    methods get the same samples so calibration data is not a confounder.
    """
    texts = load_texts(spec.dataset, None, spec.split, spec.text_field)
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)

    picked = []
    for i in order:
        t = texts[i].strip()
        if len(t.split()) < min_words:
            continue
        picked.append(t)
        if len(picked) >= spec.n_samples:
            break

    if len(picked) < spec.n_samples:
        log.warning("only found %d/%d calibration samples", len(picked), spec.n_samples)
    return picked
