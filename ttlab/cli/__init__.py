"""Command line interface for tiny-transformer-lab."""

from __future__ import annotations
import sys
from typing import Sequence

# Пытаемся подтянуть Typer-приложение из app.py
try:
    from .cli_app import app as _app  # Typer.Typer
except Exception as e:
    _app = None
    _import_error = e

def run(argv: Sequence[str] | None = None) -> int:
    """
    Совместимая с тестами точка входа.
    Делегирует в Typer app, поддерживает синтаксис 'group:cmd'.
    Возвращает exit code (int).
    """
    if _app is None:
        # Нет нового CLI — подсказываем, что именно не так
        raise RuntimeError(
            "ttlab.cli: Typer app not found (expected ttlab/cli/app.py with `app = Typer()`)."
        ) from _import_error

    args = list(sys.argv[1:] if argv is None else argv)

    # Преобразуем 'data:validate' -> ['data','validate'] (и т.п.)
    if args and ":" in args[0]:
        head, tail = args[0].split(":", 1)
        args = [head, tail] + args[1:]

    try:
        # Вызываем Click/Typer без standalone_mode, чтобы поймать код возврата
        _app(standalone_mode=False, args=args, prog_name="ttlab")
        return 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1

__all__ = ["run"]
