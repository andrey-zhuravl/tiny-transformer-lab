"""Command line interface for tiny-transformer-lab."""
import os
from pathlib import Path
import yaml

from ttlab.cli.cli_process import process_app
from ttlab.cli.cli_tokenizer import tokenizer_app
from ttlab.cli.cli_validate import validate_app
from ttlab.cli.main import main_app
from ttlab.utils.paths import get_project_path


def _init_mlflow_env() -> None:
    print("Initializing MLflow environment...")
    cfg_path = get_project_path("conf/mlflow.yaml")
    print("cfg = ", cfg_path.name)

    if not cfg_path.exists():
        print("if not cfg_path.exists():...")
        return
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    print("os.environ.setdefault...")
    # установить переменные окружения, если они не заданы
    os.environ.setdefault("MLFLOW_TRACKING_URI", cfg.get("tracking_uri", "http://127.0.0.1:5000"))
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", cfg.get("s3_endpoint_url", "http://127.0.0.1:9000"))
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", cfg.get("s3_endpoint_url", "http://127.0.0.1:9000"))
    os.environ.setdefault("AWS_ACCESS_KEY_ID", cfg.get("aws_access_key_id"))
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", cfg.get("aws_secret_access_key"))
    os.environ.setdefault("AWS_DEFAULT_REGION", cfg.get("aws_region", "us-east-1"))
    # os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    # os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9000")
    # os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9000")
    # os.environ.setdefault("AWS_ACCESS_KEY_ID", "admin")
    # os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "adminadmin")
    # os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("AWS_S3_ADDRESSING_STYLE", "path")
    print("End Initializing MLflow environment...")

# вызвать сразу при импортировании CLI
_init_mlflow_env()

__all__ = [
    "main",
    "cli_process",
    "cli_validate",
    "cli_tokenizer",
    "main_app",
    "tokenizer_app",
    "validate_app",
    "process_app",
    "_init_mlflow_env"
]
