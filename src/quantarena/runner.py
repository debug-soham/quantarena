"""Run the stages of one experiment and write results to ``<output_dir>/<run_id>/``.

    config.yaml       fully resolved config
    env.json          package versions + GPU
    ppl.json          perplexity
    memory.json       memory + disk size
    throughput.json   generation throughput

A stage is skipped when its result file exists and was produced by the same
config fingerprint, so an interrupted matrix can just be re-run.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import env
from .config import ExperimentConfig, dump_yaml

log = logging.getLogger(__name__)

STAGES = ("quantize", "ppl", "memory", "throughput")


def _stage_fn(stage: str) -> Callable[[ExperimentConfig], dict]:
    if stage == "ppl":
        from .eval.perplexity import evaluate_perplexity

        return evaluate_perplexity
    if stage == "memory":
        from .bench.memory import measure_memory

        return measure_memory
    if stage == "throughput":
        from .bench.throughput import measure_throughput

        return measure_throughput
    raise ValueError(f"unknown stage {stage!r}")


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def is_done(path: Path, fingerprint: str) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("fingerprint") == fingerprint
    except json.JSONDecodeError:
        return False


def write_result(path: Path, cfg: ExperimentConfig, stage: str, result: dict, elapsed: float) -> None:
    payload = {
        "run_id": cfg.run_id,
        "stage": stage,
        "fingerprint": cfg.fingerprint(),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage_time_s": round(elapsed, 1),
        "result": result,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def run_experiment(cfg: ExperimentConfig, stages=STAGES, force: bool = False) -> Path:
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(cfg, run_dir / "config.yaml")
    (run_dir / "env.json").write_text(json.dumps(env.collect(), indent=2))

    if "quantize" in stages:
        from .quantize import quantize

        quantize(cfg, force=force)
        _free_gpu()

    fp = cfg.fingerprint()
    for stage in stages:
        if stage == "quantize":
            continue
        out = run_dir / f"{stage}.json"
        if not force and is_done(out, fp):
            log.info("[%s] %s already done, skipping", cfg.run_id, stage)
            continue
        if cfg.checkpoint_path is not None and not cfg.checkpoint_path.exists():
            raise FileNotFoundError(
                f"{cfg.checkpoint_path} does not exist, run the quantize stage first"
            )

        log.info("[%s] running %s", cfg.run_id, stage)
        t0 = time.perf_counter()
        result = _stage_fn(stage)(cfg)
        write_result(out, cfg, stage, result, time.perf_counter() - t0)
        _free_gpu()

    return run_dir
