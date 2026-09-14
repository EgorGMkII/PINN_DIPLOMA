"""Command-line entry points for validation, parity, training, and benchmarking."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diploma-pinn")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate-data", "parity", "train", "benchmark", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    raise NotImplementedError("dispatch only after each command contract is implemented")
