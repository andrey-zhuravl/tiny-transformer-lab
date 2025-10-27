from typer.testing import CliRunner
from ttl.cli.main import app


def test_validation_run_integration() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["validation", "run", "--a", "a", "--b", "b"])
    assert result.exit_code == 0
    assert "ok" in result.stdout


def test_validation_spec_integration() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["validation", "spec", "--a", "a", "--b", "b"])
    assert result.exit_code == 0
    assert "ok" in result.stdout
