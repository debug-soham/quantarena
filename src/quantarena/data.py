from __future__ import annotations

from .config import EvalSpec


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
