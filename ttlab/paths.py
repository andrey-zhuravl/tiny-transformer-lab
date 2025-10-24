from pathlib import Path


def get_project_path(config_file_name: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return project_root / config_file_name
