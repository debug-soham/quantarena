from pathlib import Path

import pytest

from quantarena import config as C

ROOT = Path(__file__).resolve().parents[1] / "configs"


def test_deep_merge_nested():
    a = {"x": 1, "eval": {"seq_len": 2048, "stride": 512}}
    b = {"eval": {"stride": 256}, "y": 2}
    out = C.deep_merge(a, b)
    assert out == {"x": 1, "y": 2, "eval": {"seq_len": 2048, "stride": 256}}
    assert a["eval"]["stride"] == 512  # not mutated


def test_parse_override_types():
    assert C.parse_override("eval.seq_len=4096") == {"eval": {"seq_len": 4096}}
    assert C.parse_override("bench.batch_sizes=[1,2]") == {"bench": {"batch_sizes": [1, 2]}}
    assert C.parse_override("eval.max_windows=null") == {"eval": {"max_windows": None}}
    with pytest.raises(ValueError):
        C.parse_override("eval.seq_len")


def test_lists_all_models_and_methods():
    assert len(C.list_models(ROOT)) == 6
    assert set(C.list_methods(ROOT)) == {"bf16", "awq-w4g128", "gptq-w4g128", "fp8"}


@pytest.mark.parametrize("model", C.list_models(ROOT))
@pytest.mark.parametrize("method", C.list_methods(ROOT))
def test_every_combination_composes(model, method):
    cfg = C.compose(model, method, root=ROOT)
    assert cfg.model.name == model
    assert cfg.run_id.startswith(model + "__")


def test_awq_forces_fp16():
    cfg = C.compose("llama-3.1-8b", "awq-w4g128", root=ROOT)
    assert cfg.model.dtype == "float16"
    assert cfg.quant.label == "awq-w4g128"
    assert cfg.model_path.replace("\\", "/") == "checkpoints/llama-3.1-8b__awq-w4g128"


def test_baseline_loads_from_hub():
    cfg = C.compose("qwen3-8b", "bf16", root=ROOT)
    assert cfg.checkpoint_path is None
    assert cfg.model_path == "Qwen/Qwen3-8B-Base"
    assert cfg.run_id == "qwen3-8b__bf16"


def test_overrides_change_fingerprint_but_not_quant_fingerprint():
    a = C.compose("mistral-7b-v0.3", "gptq-w4g128", root=ROOT)
    b = C.compose("mistral-7b-v0.3", "gptq-w4g128", ["eval.stride=256"], root=ROOT)
    assert a.fingerprint() != b.fingerprint()
    assert a.quant_fingerprint() == b.quant_fingerprint()

    c = C.compose("mistral-7b-v0.3", "gptq-w4g128", ["calib.n_samples=256"], root=ROOT)
    assert a.quant_fingerprint() != c.quant_fingerprint()


def test_output_dir_not_in_fingerprint():
    a = C.compose("olmo-3-7b", "fp8", root=ROOT)
    b = C.compose("olmo-3-7b", "fp8", ["output_dir=elsewhere"], root=ROOT)
    assert a.fingerprint() == b.fingerprint()


@pytest.mark.parametrize(
    "override, match",
    [
        ("quant.bits=3", "AutoAWQ"),
        ("eval.stride=4096", "stride"),
        ("eval.typo=1", "unknown keys"),
        ("bogus=1", "unknown top-level"),
    ],
)
def test_validation_errors(override, match):
    with pytest.raises(ValueError, match=match):
        C.compose("llama-3.1-8b", "awq-w4g128", [override], root=ROOT)


def test_seq_len_cannot_exceed_model_context():
    with pytest.raises(ValueError, match="max_seq_len"):
        C.compose("gemma-2-9b", "bf16", ["eval.seq_len=8192", "eval.stride=512"], root=ROOT)


def test_extends(tmp_path):
    (tmp_path / "parent.yaml").write_text("seed: 1\nmodel: {name: m, hf_id: org/m}\n")
    (tmp_path / "child.yaml").write_text("extends: parent.yaml\nseed: 7\n")
    cfg = C.load_experiment(tmp_path / "child.yaml")
    assert cfg.seed == 7
    assert cfg.model.hf_id == "org/m"
