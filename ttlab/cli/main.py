import typer
from ttlab.cli.cli_process import process_app
from ttlab.cli.cli_tokenizer import tokenizer_app
from ttlab.cli.cli_validate import validate_app
from ttlab.cli.model_cli import model_app

main_app = typer.Typer(help="Tiny Transformer Lab CLI")

# Подключаем доменные группы как подкоманды верхнего уровня:
main_app.add_typer(tokenizer_app, name="tokenizer")
main_app.add_typer(validate_app, name="validate")
main_app.add_typer(process_app, name="process")
main_app.add_typer(model_app, name="model")

def main() -> None:
    main_app()

__all__ = [
    "main",
    "main_app",
    "tokenizer_app",
    "validate_app",
    "process_app",
    "model_app",
]