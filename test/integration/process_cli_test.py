from typer.testing import CliRunner
from ttl.cli.main import app


def test_process_run_integration() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["process", "run", "--a", "a", "--b", "b"])
    assert result.exit_code == 0
    assert "ok" in result.stdout


def test_process_spec_integration() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["process", "spec", "--a", "a", "--b", "b"])
    assert result.exit_code == 0
    assert "ok" in result.stdout
