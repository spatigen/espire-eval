from pathlib import Path


def proj_dir() -> Path:
    return Path(__file__).parent.parent
