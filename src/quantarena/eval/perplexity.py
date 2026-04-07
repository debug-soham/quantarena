"""Sliding-window perplexity.

Each window of ``seq_len`` tokens advances by ``stride``; only the tokens that
were not scored by the previous window contribute to the loss, so every token
after the first is scored exactly once with at least ``seq_len - stride`` tokens
of context (except at the very start).
"""

from __future__ import annotations

import math
import time

from ..config import ExperimentConfig


def sliding_window_nll(model, input_ids, seq_len: int, stride: int, max_windows: int | None = None):
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    n = input_ids.size(1)
    nll_sum = 0.0
    n_tokens = 0
    windows = 0
    prev_end = 0

    # After the first window, advance by at most seq_len - 1 so the first new target
    # always has at least one token of context inside the window. With stride == seq_len
    # the naive loop silently drops one token per window.
    step = min(stride, seq_len - 1)

    while prev_end < n:
        if prev_end == 0:
            begin, end = 0, min(seq_len, n)
        else:
            end = min(prev_end + step, n)
            begin = max(0, end - seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(device)
        labels = ids.clone()
        labels[:, :-trg_len] = -100

        with torch.no_grad():
            logits = model(ids).logits

        # Do the reduction ourselves in fp32 instead of relying on model-specific loss code.
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        nll_sum += loss.item()
        n_tokens += int((shift_labels != -100).sum().item())
        windows += 1
        prev_end = end

        if end == n or (max_windows and windows >= max_windows):
            break

    return nll_sum, n_tokens, windows


def evaluate_perplexity(cfg: ExperimentConfig) -> dict:
    from ..data import eval_corpus
    from ..loading import load_model

    model, tok = load_model(cfg.model_path, cfg.model)
    text = eval_corpus(cfg.eval)
    input_ids = tok(text, return_tensors="pt").input_ids

    t0 = time.perf_counter()
    nll, n_tokens, windows = sliding_window_nll(
        model, input_ids, cfg.eval.seq_len, cfg.eval.stride, cfg.eval.max_windows
    )
    elapsed = time.perf_counter() - t0

    return {
        "perplexity": math.exp(nll / n_tokens),
        "nll_per_token": nll / n_tokens,
        "tokens_scored": n_tokens,
        "corpus_tokens": int(input_ids.size(1)),
        "windows": windows,
        "seq_len": cfg.eval.seq_len,
        "stride": cfg.eval.stride,
        "dataset": f"{cfg.eval.dataset}/{cfg.eval.subset}/{cfg.eval.split}",
        "elapsed_s": round(elapsed, 2),
    }
