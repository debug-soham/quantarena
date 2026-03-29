from __future__ import annotations

import argparse
import logging
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
