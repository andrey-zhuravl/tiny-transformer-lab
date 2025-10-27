import typer
from ttl.cli import process as process_cli
from ttl.cli import validation as validation_cli

app = typer.Typer(help="ttl CLI")
app.add_typer(process_cli.app, name="process")
app.add_typer(validation_cli.app, name="validation")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
