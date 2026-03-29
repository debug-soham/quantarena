"""Record the software/hardware environment next to every result."""

from __future__ import annotations

import platform
import sys
from importlib import metadata

PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "autoawq",
    "gptqmodel",
    "llmcompressor",
    "compressed-tensors",
    "vllm",
    "quantarena",
)


def package_versions() -> dict[str, str | None]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def gpu_info() -> list[dict]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append(
            {
                "name": p.name,
                "memory_gib": round(p.total_memory / 1024**3, 1),
                "capability": f"{p.major}.{p.minor}",
            }
        )
    return gpus


def collect() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpus": gpu_info(),
    }
    try:
        import torch

        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    except ImportError:
        pass
    return info
