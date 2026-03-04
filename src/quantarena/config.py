"""Experiment configuration.

An experiment is built by merging YAML fragments in order:

    configs/base.yaml -> configs/models/<model>.yaml -> configs/methods/<method>.yaml -> --set overrides

Every fragment has the same shape as the final config, so any file can override
any field.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

import yaml

METHODS = ("none", "awq", "gptq", "fp8")


def config_root() -> Path:
    return Path(os.environ.get("QUANTARENA_CONFIGS", "configs"))


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent:
        data = deep_merge(load_yaml(path.parent / parent), data)
    return data


def parse_override(expr: str) -> dict:
    """Turn ``eval.seq_len=4096`` into ``{"eval": {"seq_len": 4096}}``."""
    key, sep, raw = expr.partition("=")
    if not sep or not key:
        raise ValueError(f"override must look like key=value, got {expr!r}")
    value: Any = yaml.safe_load(raw)
    for part in reversed(key.strip().split(".")):
        value = {part: value}
    return value


@dataclass
class ModelSpec:
    name: str
    hf_id: str
    revision: str | None = None
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    max_seq_len: int = 4096


@dataclass
class QuantSpec:
    method: str = "none"
    bits: int = 16
    group_size: int = 128
    options: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.method == "none":
            return "bf16"
        if self.method == "fp8":
            return "fp8"
        return f"{self.method}-w{self.bits}g{self.group_size}"


@dataclass
class CalibSpec:
    dataset: str = "mit-han-lab/pile-val-backup"
    split: str = "validation"
    text_field: str = "text"
    n_samples: int = 128
    seq_len: int = 512


@dataclass
class EvalSpec:
    dataset: str = "wikitext"
    subset: str | None = "wikitext-2-raw-v1"
    split: str = "test"
    text_field: str = "text"
    seq_len: int = 2048
    stride: int = 512
    max_windows: int | None = None


@dataclass
class BenchSpec:
    backend: str = "vllm"
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8, 32])
    input_len: int = 512
    output_len: int = 128
    warmup: int = 2
    iters: int = 5
    gpu_memory_utilization: float = 0.85


def _build(kls, raw: dict | None, section: str):
    raw = raw or {}
    known = {f.name for f in fields(kls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown keys in '{section}': {sorted(unknown)}")
    return kls(**raw)


@dataclass
class ExperimentConfig:
    model: ModelSpec
    quant: QuantSpec = field(default_factory=QuantSpec)
    calib: CalibSpec = field(default_factory=CalibSpec)
    eval: EvalSpec = field(default_factory=EvalSpec)
    bench: BenchSpec = field(default_factory=BenchSpec)
    seed: int = 0
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        data = dict(data)
        if not data.get("model"):
            raise ValueError("config has no 'model' section")
        sections = {
            "model": ModelSpec,
            "quant": QuantSpec,
            "calib": CalibSpec,
            "eval": EvalSpec,
            "bench": BenchSpec,
        }
        kwargs = {name: _build(kls, data.pop(name, None), name) for name, kls in sections.items()}
        top = {f.name for f in fields(cls)} - set(sections)
        unknown = set(data) - top
        if unknown:
            raise ValueError(f"unknown top-level keys: {sorted(unknown)}")
        cfg = cls(**kwargs, **data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        q = self.quant
        if q.method not in METHODS:
            raise ValueError(f"unknown quant method {q.method!r}, expected one of {METHODS}")
        if q.method == "awq" and q.bits != 4:
            raise ValueError("AutoAWQ only supports 4-bit weights")
        if q.method == "gptq" and q.bits not in (2, 3, 4, 8):
            raise ValueError(f"GPTQ bits must be 2/3/4/8, got {q.bits}")
        if q.method == "fp8" and q.bits != 8:
            raise ValueError("fp8 method needs bits=8")
        if self.eval.stride > self.eval.seq_len:
            raise ValueError("eval.stride cannot be larger than eval.seq_len")
        if self.eval.seq_len > self.model.max_seq_len:
            raise ValueError(
                f"eval.seq_len={self.eval.seq_len} exceeds model.max_seq_len={self.model.max_seq_len}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def run_id(self) -> str:
        return f"{self.model.name}__{self.quant.label}"

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_id

    @property
    def checkpoint_path(self) -> Path | None:
        if self.quant.method == "none":
            return None
        return Path(self.checkpoint_dir) / self.run_id

    @property
    def model_path(self) -> str:
        """What to load for evaluation: the quantized checkpoint, or the HF id for the baseline."""
        ckpt = self.checkpoint_path
        return str(ckpt) if ckpt is not None else self.model.hf_id

    def fingerprint(self) -> str:
        d = self.to_dict()
        d.pop("output_dir")
        d.pop("checkpoint_dir")
        return _hash(d)

    def quant_fingerprint(self) -> str:
        """Hash of everything that affects the quantized weights (not eval/bench settings)."""
        d = self.to_dict()
        return _hash({k: d[k] for k in ("model", "quant", "calib", "seed")})


def _hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _find(directory: Path, name: str) -> Path:
    for suffix in (".yaml", ".yml"):
        p = directory / f"{name}{suffix}"
        if p.exists():
            return p
    available = ", ".join(sorted(p.stem for p in directory.glob("*.y*ml")))
    raise FileNotFoundError(f"no config '{name}' in {directory} (available: {available})")


def list_models(root: str | Path | None = None) -> list[str]:
    root = Path(root) if root else config_root()
    return sorted(p.stem for p in (root / "models").glob("*.y*ml"))


def list_methods(root: str | Path | None = None) -> list[str]:
    root = Path(root) if root else config_root()
    return sorted(p.stem for p in (root / "methods").glob("*.y*ml"))


def compose(
    model: str,
    method: str,
    overrides: Iterable[str] = (),
    root: str | Path | None = None,
) -> ExperimentConfig:
    root = Path(root) if root else config_root()
    data = load_yaml(root / "base.yaml")
    data = deep_merge(data, load_yaml(_find(root / "models", model)))
    data = deep_merge(data, load_yaml(_find(root / "methods", method)))
    for expr in overrides:
        data = deep_merge(data, parse_override(expr))
    return ExperimentConfig.from_dict(data)


def load_experiment(path: str | Path, overrides: Iterable[str] = ()) -> ExperimentConfig:
    data = load_yaml(path)
    for expr in overrides:
        data = deep_merge(data, parse_override(expr))
    return ExperimentConfig.from_dict(data)


def dump_yaml(cfg: ExperimentConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
