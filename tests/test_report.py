import json

import yaml

from quantarena import report


def _write_run(root, model, quant, ppl, weights, tok_s):
    method = quant["method"]
    name = f"{model}__{method}"
    d = root / name
    d.mkdir()
    (d / "config.yaml").write_text(
        yaml.safe_dump({"model": {"name": model}, "quant": {"group_size": 128, **quant}})
    )

    def dump(stage, result):
        (d / f"{stage}.json").write_text(json.dumps({"fingerprint": "x", "result": result}))

    dump("ppl", {"perplexity": ppl})
    dump("memory", {"weights_gib": weights, "forward_peak_gib": weights + 1, "disk_gib": weights})
    dump(
        "throughput",
        {"results": [{"batch_size": 1, "output_tok_s": tok_s, "ttft_ms": 10.0}, {"batch_size": 32, "oom": True}]},
    )


def test_drift_against_baseline(tmp_path):
    _write_run(tmp_path, "m", {"method": "none", "bits": 16}, 6.0, 16.0, 50.0)
    _write_run(tmp_path, "m", {"method": "awq", "bits": 4}, 6.3, 5.0, 100.0)
    _write_run(tmp_path, "m", {"method": "fp8", "bits": 8}, 6.06, 8.0, 75.0)

    rows = report.build(tmp_path)
    assert [r["method"] for r in rows] == ["bf16", "fp8", "awq-w4g128"]

    by = {r["method"]: r for r in rows}
    assert by["bf16"]["ppl_drift_pct"] == 0.0
    assert by["awq-w4g128"]["ppl_drift_pct"] == 5.0
    assert by["fp8"]["ppl_drift_pct"] == 1.0
    assert by["fp8"]["mem_ratio"] == 0.5
    assert by["awq-w4g128"]["speedup_bs1"] == 2.0
    assert by["awq-w4g128"]["tok_s_bs32"] == "oom"
    assert "speedup_bs32" not in by["awq-w4g128"]


def test_missing_baseline_leaves_drift_empty(tmp_path):
    _write_run(tmp_path, "m", {"method": "gptq", "bits": 4}, 7.0, 5.0, 90.0)
    rows = report.build(tmp_path)
    assert "ppl_drift_pct" not in rows[0]


def test_markdown_table(tmp_path):
    _write_run(tmp_path, "m", {"method": "none", "bits": 16}, 6.0, 16.0, 50.0)
    _write_run(tmp_path, "m", {"method": "awq", "bits": 4}, 6.3, 5.0, 100.0)
    md = report.to_markdown(report.build(tmp_path))
    lines = md.strip().splitlines()
    assert lines[0].startswith("| model | method | wikitext2 ppl | drift %")
    assert "tok/s bs1" in lines[0] and "tok/s bs32" in lines[0]
    assert len(lines) == 4
    assert lines[3].startswith("|  | awq-w4g128 | 6.300 | 5.00 |")
    assert "oom" in lines[3]


def test_csv_roundtrip(tmp_path):
    _write_run(tmp_path, "m", {"method": "none", "bits": 16}, 6.0, 16.0, 50.0)
    rows = report.build(tmp_path)
    out = tmp_path / "s.csv"
    report.write_csv(rows, out)
    header = out.read_text().splitlines()[0].split(",")
    assert header[:3] == ["model", "method", "ppl"]
