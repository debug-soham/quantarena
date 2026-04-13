from __future__ import annotations

import argparse
import logging
import subprocess
import sys

import yaml

from . import config as C


def _add_experiment_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="model config name, e.g. llama-3.1-8b")
    p.add_argument("method", help="method config name, e.g. awq-w4g128")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config field, e.g. --set eval.seq_len=4096 (repeatable)",
    )


def _compose(args) -> C.ExperimentConfig:
    return C.compose(args.model, args.method, args.overrides, root=args.config_root)


def cmd_list(args) -> int:
    print("models:")
    for m in C.list_models(args.config_root):
        print(f"  {m}")
    print("methods:")
    for m in C.list_methods(args.config_root):
        print(f"  {m}")
    return 0


def cmd_show(args) -> int:
    cfg = _compose(args)
    print(f"# run_id: {cfg.run_id}  fingerprint: {cfg.fingerprint()}")
    print(yaml.safe_dump(cfg.to_dict(), sort_keys=False), end="")
    return 0


def cmd_quantize(args) -> int:
    from .quantize import quantize

    cfg = _compose(args)
    quantize(cfg, force=args.force)
    return 0


def _parse_stages(value: str) -> tuple[str, ...]:
    from .runner import STAGES

    stages = tuple(s.strip() for s in value.split(",") if s.strip())
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown stage(s) {bad}, choose from {STAGES}")
    return stages


def cmd_run(args) -> int:
    from .runner import run_experiment

    cfg = _compose(args)
    run_dir = run_experiment(cfg, stages=args.stages, force=args.force)
    print(run_dir)
    return 0


def cmd_matrix(args) -> int:
    """Run every model x method, one subprocess per stage.

    Separate processes keep vLLM and transformers from fighting over GPU memory and
    make sure one crash (OOM, missing kernel) does not take down the whole sweep.
    """
    models = args.models or C.list_models(args.config_root)
    methods = args.methods or C.list_methods(args.config_root)

    # validate everything up front so a typo doesn't show up 6 hours in
    for model in models:
        for method in methods:
            C.compose(model, method, args.overrides, root=args.config_root)

    base = [sys.executable, "-m", "quantarena"]
    if args.config_root:
        base += ["--config-root", args.config_root]

    failures = []
    for model in models:
        for method in methods:
            for stage in args.stages:
                cmd = base + ["run", model, method, "--stages", stage]
                for o in args.overrides:
                    cmd += ["--set", o]
                if args.force:
                    cmd.append("--force")
                print("$ " + " ".join(cmd[2:]), flush=True)
                if args.dry_run:
                    continue
                rc = subprocess.call(cmd)
                if rc != 0:
                    failures.append((model, method, stage, rc))
                    if not args.keep_going:
                        print(f"failed with exit code {rc}", file=sys.stderr)
                        return rc
                    if stage == "quantize":
                        break  # nothing to evaluate without a checkpoint

    if failures:
        print("\nfailed:", file=sys.stderr)
        for model, method, stage, rc in failures:
            print(f"  {model} {method} {stage} (exit {rc})", file=sys.stderr)
        return 1
    return 0


def cmd_report(args) -> int:
    from . import report

    rows = report.build(args.results)
    if not rows:
        print(f"no results found in {args.results}")
        return 1
    out = args.csv or f"{args.results}/summary.csv"
    report.write_csv(rows, out)
    print(f"wrote {len(rows)} rows to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantarena")
    parser.add_argument("--config-root", default=None, help="directory with base.yaml, models/, methods/")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list available models and methods")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="print the resolved config for an experiment")
    _add_experiment_args(p)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("quantize", help="produce a quantized checkpoint")
    _add_experiment_args(p)
    p.add_argument("--force", action="store_true", help="re-quantize even if the checkpoint is current")
    p.set_defaults(func=cmd_quantize)

    p = sub.add_parser("run", help="quantize (if needed) and run eval/bench stages")
    _add_experiment_args(p)
    p.add_argument(
        "--stages",
        type=_parse_stages,
        default="quantize,ppl,memory,throughput",
        help="comma separated subset of quantize,ppl,memory,throughput",
    )
    p.add_argument("--force", action="store_true", help="re-run stages even if results exist")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("matrix", help="run all model x method combinations")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--stages", type=_parse_stages, default="quantize,ppl,memory,throughput")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-going", action="store_true", help="continue after a failed run")
    p.add_argument("--dry-run", action="store_true", help="only print the commands")
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser("report", help="aggregate results into a summary table")
    p.add_argument("--results", default="results")
    p.add_argument("--csv", default=None, help="output path (default: <results>/summary.csv)")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
