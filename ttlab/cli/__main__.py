"""Module entry-point for ``python -m ttlab.cli``."""

from __future__ import annotations

import sys

from . import run


def main() -> None:  # pragma: no cover - thin wrapper around command runner
    exit_code = run()
    sys.exit(exit_code.value)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
