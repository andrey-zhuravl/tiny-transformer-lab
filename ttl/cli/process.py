import typer
from ttl.core import process as core_process

app = typer.Typer(help="Process commands")


@app.command("run")
def run(
    a: str = typer.Option(..., "--a", help="First arg"),
    b: str = typer.Option(..., "--b", help="Second arg"),
) -> None:
    res = core_process.run(a, b)
    typer.echo(res)


@app.command("spec")
def spec(
    a: str = typer.Option(..., "--a", help="First arg"),
    b: str = typer.Option(..., "--b", help="Second arg"),
) -> None:
    res = core_process.spec(a, b)
    typer.echo(res)
