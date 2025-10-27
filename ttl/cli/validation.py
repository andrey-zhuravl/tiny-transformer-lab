import typer
from ttl.core import validation as core_validation

app = typer.Typer(help="Validation commands")


@app.command("run")
def run(
    a: str = typer.Option(..., "--a", help="First arg"),
    b: str = typer.Option(..., "--b", help="Second arg"),
) -> None:
    res = core_validation.run(a, b)
    typer.echo(res)


@app.command("spec")
def spec(
    a: str = typer.Option(..., "--a", help="First arg"),
    b: str = typer.Option(..., "--b", help="Second arg"),
) -> None:
    res = core_validation.spec(a, b)
    typer.echo(res)
