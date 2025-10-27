from __future__ import annotations

import json
from pathlib import Path

import typer

from ..core.tokenizer import inspect_tokenizer, train_or_import_tokenizer


class ExitCode(int):
    OK = 0
    INVALID_INPUT = 2
    IO_ERROR = 3
    UNKNOWN = 4

app = typer.Typer(help="Tokenizer utilities")

@app.command("train")
def tok_train(
    dataset_manifest: Path = typer.Option(..., "--dataset-manifest", help="Path to dataset.manifest.json"),
    algo: str = typer.Option(..., "--algo", help="Tokenization algorithm: char|bpe|unigram"),
    vocab_size: int = typer.Option(8000, "--vocab-size", help="Target vocabulary size"),
    norm: str = typer.Option("nfc", "--norm", help="Text normalization: none|nfc"),
    lower: bool = typer.Option(True, "--lower/--no-lower", help="Toggle lower casing"),
    punct_policy: str = typer.Option("keep", "--punct-policy", help="keep|strip|space"),
    seed: int = typer.Option(13, "--seed", help="Deterministic random seed"),
    out_dir: Path = typer.Option(Path("out/tokenizer"), "--out", help="Output directory"),
    use_external_tokenizer: Path | None = typer.Option(None, "--use-external-tokenizer", help="Import an existing tokenizer.json"),
) -> ExitCode:
    try:
        result = train_or_import_tokenizer(
            dataset_manifest=dataset_manifest,
            algo=algo,
            vocab_size=vocab_size,
            norm=norm,
            lower=lower,
            punct_policy=punct_policy,
            seed=seed,
            out_dir=out_dir,
            external_tokenizer=use_external_tokenizer,
        )
    except FileNotFoundError as exc:  # pragma: no cover - exercised via CLI integration tests
        typer.echo(f"[IO] {exc}")
        raise typer.Exit(code=ExitCode.IO_ERROR)
    except ValueError as exc:  # pragma: no cover
        typer.echo(f"[INVALID] {exc}")
        raise typer.Exit(code=ExitCode.INVALID_INPUT)
    except Exception as exc:  # pragma: no cover - ensures stable CLI exit codes
        typer.echo(f"[UNKNOWN] {exc}")
        raise typer.Exit(code=ExitCode.UNKNOWN)

    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    return ExitCode.OK


@app.command("inspect")
def tok_inspect(
    tokenizer_path: Path = typer.Option(..., "--tokenizer", help="Path to tokenizer.json"),
    dataset_manifest: Path = typer.Option(..., "--dataset-manifest", help="Dataset manifest"),
    out_dir: Path = typer.Option(Path("out/tokenizer"), "--out", help="Output directory"),
) -> None:
    try:
        report = inspect_tokenizer(tokenizer_path=tokenizer_path, dataset_manifest=dataset_manifest, out_dir=out_dir)
    except FileNotFoundError as exc:  # pragma: no cover
        typer.echo(f"[IO] {exc}")
        raise typer.Exit(code=ExitCode.IO_ERROR)
    except ValueError as exc:  # pragma: no cover
        typer.echo(f"[INVALID] {exc}")
        raise typer.Exit(code=ExitCode.INVALID_INPUT)
    except Exception as exc:  # pragma: no cover
        typer.echo(f"[UNKNOWN] {exc}")
        raise typer.Exit(code=ExitCode.UNKNOWN)

    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
