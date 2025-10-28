"""tiny-transformer-lab core package."""
from . import cli  # импортируем подпакет, чтобы он реально был доступен как ttlab.cli
from . import core, utils  # раскомментируй, когда подпакеты появятся

__all__ = ["cli", "core", "utils"]  # добавляй "core", "utils" когда они реально есть
__version__ = "0.1.0"