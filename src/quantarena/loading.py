from __future__ import annotations

from .config import ModelSpec


def torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_tokenizer(path: str, spec: ModelSpec):
    from transformers import AutoTokenizer

    kwargs = {"trust_remote_code": spec.trust_remote_code}
    if path == spec.hf_id and spec.revision:
        kwargs["revision"] = spec.revision
    return AutoTokenizer.from_pretrained(path, **kwargs)


def load_model(path: str, spec: ModelSpec, device_map: str = "cuda"):
    """Load a baseline or quantized checkpoint through transformers.

    AWQ, GPTQ and compressed-tensors FP8 checkpoints all carry a quantization_config,
    so transformers dispatches to the right kernels as long as the backend is installed.
    """
    from transformers import AutoModelForCausalLM

    kwargs = {
        "torch_dtype": torch_dtype(spec.dtype),
        "device_map": device_map,
        "trust_remote_code": spec.trust_remote_code,
    }
    if spec.attn_implementation:
        kwargs["attn_implementation"] = spec.attn_implementation
    if path == spec.hf_id and spec.revision:
        kwargs["revision"] = spec.revision

    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.eval()
    return model, load_tokenizer(path, spec)
