import typer
from .cli_process import app as process_app
from .cli_tokenizer import app as tok_app
from .cli_validate import app as validate_app

app = typer.Typer(help="Tiny Transformer Lab CLI")

# Подключаем доменные группы как подкоманды верхнего уровня:
app.add_typer(process_app, name="process")
app.add_typer(tok_app, name="tok")
app.add_typer(validate_app, name="validate")

def main() -> None:
    app()

if __name__ == "__main__":
    main()