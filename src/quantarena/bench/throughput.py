"""Generation throughput.

Prompts are random token ids of fixed length and generation is forced to exactly
``output_len`` tokens, so every method does the same amount of work.

Reported per batch size:
  ttft_ms       time to first token (prefill + 1 decode step) for the whole batch
  e2e_s         wall time for the full generation
  output_tok_s  batch * output_len / e2e_s
  decode_tok_s  batch * (output_len - 1) / (e2e_s - ttft)
"""

from __future__ import annotations

import logging
import random
import statistics
import time

from ..config import ExperimentConfig

log = logging.getLogger(__name__)


def random_prompts(n: int, length: int, vocab_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    # stay clear of the low ids where most special tokens live
    lo = min(1000, vocab_size // 10)
    return [[rng.randrange(lo, vocab_size) for _ in range(length)] for _ in range(n)]


def summarize(batch_size: int, output_len: int, ttfts: list[float], e2es: list[float]) -> dict:
    ttft = statistics.median(ttfts)
    e2e = statistics.median(e2es)
    decode_time = max(e2e - ttft, 1e-9)
    return {
        "batch_size": batch_size,
        "ttft_ms": round(ttft * 1000, 2),
        "e2e_s": round(e2e, 4),
        "output_tok_s": round(batch_size * output_len / e2e, 2),
        "decode_tok_s": round(batch_size * (output_len - 1) / decode_time, 2),
        "iters": len(e2es),
    }


def bench_hf(cfg: ExperimentConfig) -> list[dict]:
    import torch

    from ..loading import load_model

    b = cfg.bench
    model, tok = load_model(cfg.model_path, cfg.model)
    vocab = min(model.config.vocab_size, len(tok))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def timed(**kw) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(**kw)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    results = []
    for bs in b.batch_sizes:
        ids = torch.tensor(random_prompts(bs, b.input_len, vocab, cfg.seed), device=model.device)
        common = dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            pad_token_id=pad_id,
        )
        full = dict(common, max_new_tokens=b.output_len, min_new_tokens=b.output_len)
        first = dict(common, max_new_tokens=1, min_new_tokens=1)
        try:
            for _ in range(b.warmup):
                timed(**full)
            ttfts = [timed(**first) for _ in range(b.iters)]
            e2es = [timed(**full) for _ in range(b.iters)]
        except torch.cuda.OutOfMemoryError:
            log.warning("OOM at batch size %d, stopping", bs)
            torch.cuda.empty_cache()
            results.append({"batch_size": bs, "oom": True})
            break
        r = summarize(bs, b.output_len, ttfts, e2es)
        log.info("bs=%d  %.1f tok/s  ttft %.1f ms", bs, r["output_tok_s"], r["ttft_ms"])
        results.append(r)

    del model
    torch.cuda.empty_cache()
    return results


def bench_vllm(cfg: ExperimentConfig) -> list[dict]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    b = cfg.bench
    llm = LLM(
        model=cfg.model_path,
        dtype=cfg.model.dtype,
        seed=cfg.seed,
        max_model_len=b.input_len + b.output_len,
        max_num_seqs=max(b.batch_sizes),
        gpu_memory_utilization=b.gpu_memory_utilization,
        trust_remote_code=cfg.model.trust_remote_code,
        # same random prompts are reused across iterations, caching would cheat
        enable_prefix_caching=False,
    )
    vocab = len(llm.get_tokenizer())

    full = SamplingParams(temperature=0.0, max_tokens=b.output_len, min_tokens=b.output_len, ignore_eos=True)
    first = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    def timed(prompts, params) -> float:
        t0 = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        dt = time.perf_counter() - t0
        produced = sum(len(o.outputs[0].token_ids) for o in outs)
        expected = len(prompts) * params.max_tokens
        if produced != expected:
            log.warning("expected %d output tokens, got %d", expected, produced)
        return dt

    results = []
    for bs in b.batch_sizes:
        prompts = [
            TokensPrompt(prompt_token_ids=p) for p in random_prompts(bs, b.input_len, vocab, cfg.seed)
        ]
        for _ in range(b.warmup):
            timed(prompts, full)
        ttfts = [timed(prompts, first) for _ in range(b.iters)]
        e2es = [timed(prompts, full) for _ in range(b.iters)]
        r = summarize(bs, b.output_len, ttfts, e2es)
        log.info("bs=%d  %.1f tok/s  ttft %.1f ms", bs, r["output_tok_s"], r["ttft_ms"])
        results.append(r)

    return results


BACKENDS = {"hf": bench_hf, "vllm": bench_vllm}


def measure_throughput(cfg: ExperimentConfig) -> dict:
    try:
        fn = BACKENDS[cfg.bench.backend]
    except KeyError:
        raise ValueError(f"unknown bench backend {cfg.bench.backend!r}") from None
    return {
        "backend": cfg.bench.backend,
        "input_len": cfg.bench.input_len,
        "output_len": cfg.bench.output_len,
        "results": fn(cfg),
    }
