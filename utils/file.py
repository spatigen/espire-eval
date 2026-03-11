from pathlib import Path


def proj_dir() -> Path:
    return Path(__file__).parent.parent


def read_prompt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        res = f.read()
    return res
