"""Aggregate per-run results into one table.

Drift columns compare each quantized run against the bf16 run of the same model:
  ppl_drift_pct   (ppl / ppl_bf16 - 1) * 100
  mem_ratio       weights_gib / weights_gib_bf16
  speedup_bsN     output_tok_s / output_tok_s_bf16 at batch size N
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

BASELINE = "bf16"


def _result(run_dir: Path, stage: str) -> dict | None:
    p = run_dir / f"{stage}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["result"]


def _label(quant: dict) -> str:
    method = quant["method"]
    if method == "none":
        return "bf16"
    if method == "fp8":
        return "fp8"
    return f"{method}-w{quant['bits']}g{quant['group_size']}"


def collect(results_dir: str | Path) -> list[dict]:
    rows = []
    for run_dir in sorted(Path(results_dir).iterdir()):
        cfg_path = run_dir / "config.yaml"
        if not run_dir.is_dir() or not cfg_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        row: dict = {
            "model": cfg["model"]["name"],
            "method": _label(cfg["quant"]),
        }

        ppl = _result(run_dir, "ppl")
        if ppl:
            row["ppl"] = ppl["perplexity"]

        mem = _result(run_dir, "memory")
        if mem:
            row["weights_gib"] = mem["weights_gib"]
            row["forward_peak_gib"] = mem["forward_peak_gib"]
            row["disk_gib"] = mem["disk_gib"]

        tput = _result(run_dir, "throughput")
        if tput:
            for r in tput["results"]:
                bs = r["batch_size"]
                if r.get("oom"):
                    row[f"tok_s_bs{bs}"] = "oom"
                    continue
                row[f"tok_s_bs{bs}"] = r["output_tok_s"]
                row[f"ttft_ms_bs{bs}"] = r["ttft_ms"]

        rows.append(row)
    return rows


def _num(x) -> float | None:
    return x if isinstance(x, (int, float)) else None


def add_drift(rows: list[dict]) -> list[dict]:
    base = {r["model"]: r for r in rows if r["method"] == BASELINE}
    for r in rows:
        b = base.get(r["model"])
        if b is None:
            continue
        if _num(r.get("ppl")) and _num(b.get("ppl")):
            r["ppl_drift_pct"] = round((r["ppl"] / b["ppl"] - 1) * 100, 3)
        if _num(r.get("weights_gib")) and _num(b.get("weights_gib")):
            r["mem_ratio"] = round(r["weights_gib"] / b["weights_gib"], 3)
        for key in list(r):
            if key.startswith("tok_s_bs"):
                cur, ref = _num(r[key]), _num(b.get(key))
                if cur and ref:
                    r["speedup_" + key.removeprefix("tok_s_")] = round(cur / ref, 3)
    return rows


def _columns(rows: list[dict]) -> list[str]:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols


def write_csv(rows: list[dict], path: str | Path) -> None:
    cols = _columns(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _fmt(v, digits: int = 2) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def to_markdown(rows: list[dict]) -> str:
    """Compact table for the README: quality, memory and the largest common batch size."""
    bss = sorted(
        {int(k.removeprefix("tok_s_bs")) for r in rows for k in r if k.startswith("tok_s_bs")}
    )
    cols = [
        ("model", "model", 0),
        ("method", "method", 0),
        ("ppl", "wikitext2 ppl", 3),
        ("ppl_drift_pct", "drift %", 2),
        ("weights_gib", "weights GiB", 2),
        ("mem_ratio", "mem ×", 2),
    ]
    for bs in (bss[:1] + bss[-1:]) if len(bss) > 1 else bss:
        cols.append((f"tok_s_bs{bs}", f"tok/s bs{bs}", 1))
        cols.append((f"speedup_bs{bs}", f"speedup bs{bs}", 2))

    lines = [
        "| " + " | ".join(h for _, h, _ in cols) + " |",
        "|" + "|".join("---" if i < 2 else "---:" for i in range(len(cols))) + "|",
    ]
    prev_model = None
    for r in rows:
        cells = []
        for key, _, digits in cols:
            if key == "model" and r["model"] == prev_model:
                cells.append("")
                continue
            cells.append(_fmt(r.get(key), digits))
        prev_model = r["model"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build(results_dir: str | Path) -> list[dict]:
    rows = add_drift(collect(results_dir))
    order = {"bf16": 0, "fp8": 1}
    rows.sort(key=lambda r: (r["model"], order.get(r["method"], 2), r["method"]))
    return rows
