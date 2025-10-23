"""Command line interface for tiny-transformer-lab."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from .data_validate import run_data_validate
from ttlab.core.validate import ExitCode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ttlab", description="tiny-transformer-lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "data:validate", help="Validate a dataset against a YAML schema."
    )
    validate_parser.add_argument("--in", dest="in_path", required=True, help="Dataset path")
    validate_parser.add_argument(
        "--schema", dest="schema_path", required=True, help="Schema YAML path"
    )
    validate_parser.add_argument(
        "--format",
        dest="data_format",
        default="JSONL",
        help="Dataset format (jsonl or parquet)",
    )
    validate_parser.add_argument(
        "--metrics-dir",
        dest="metrics_dir",
        default="out",
        help="Directory to store validation metrics",
    )

    return parser


def run(argv: Optional[Iterable[str]] = None) -> ExitCode:
    """Execute the CLI using the provided arguments."""

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "data:validate":
        exit_code = run_data_validate(
            in_path=Path(args.in_path),
            schema_path=Path(args.schema_path),
            data_format=args.data_format,
            metrics_dir=Path(args.metrics_dir) if args.metrics_dir else None,
        )
    else:  # pragma: no cover - argparse prevents reaching this branch
        parser.error(f"Unknown command: {args.command}")
        exit_code = ExitCode.UNKNOWN

    return exit_code


__all__ = ["run"]
