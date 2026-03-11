import ast
import json
from pathlib import Path
from typing import Iterator

import numpy as np


def extract_task_uid(log_content: str) -> str | None:
    """
    Extract task UID from log file content using string parsing.

    Args:
        log_content (str): Content of the log file

    Returns:
        str | None: Extracted task UID or None if not found
    """
    try:
        uid_start = log_content.find("Task selected") + 13
        if uid_start == -1:  # Not found
            return None

        uid_end = log_content.find("}", uid_start)
        task_info = json.loads(log_content[uid_start : uid_end + 1])

        return task_info["id"]
    except Exception:
        return None


def populate_task_cache(log_dir: str | Path) -> Iterator[str]:
    """
    Populate task cache from existing log files in directory.

    Parses run.log files to extract completed task UIDs for avoiding repetition.

    Args:
        log_dir (str | Path): Directory containing log files to parse

    Yields:
        str: Task UID from cache
    """
    # Convert string path to Path object if necessary
    log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir

    if not log_dir.exists() or not log_dir.is_dir():
        return

    for log_subdir in log_dir.iterdir():
        if not log_subdir.is_dir():
            continue

        log_file = log_subdir / "run.log"
        if not log_file.exists():
            continue

        try:
            log_content = log_file.read_text(encoding="utf-8")
            if "uid" in log_content:
                # Extract task UID from log content
                task_uid = extract_task_uid(log_content)
                if task_uid:
                    yield task_uid
        except Exception:
            # Skip files that can't be read or parsed
            continue


def sample_new_task(
    task_list: list[dict],
    rng: np.random.Generator,
    cache: set[str],
    max_attempts: int = 1000,
) -> dict | None:
    """
    Sample a task that hasn't been completed previously.

    Args:
        task_list (list[dict]): Available tasks from server
        rng (np.random.Generator): Random number generator
        cache (set[str]): A set of task UIDs that have already been sampled.
        max_attempts (int): Maximum number of random attempts before falling back to exhaustive search. Defaults to 1000.

    Returns:
        dict | None: Selected task or None if no new tasks available
    """
    # Primary sampling with limited attempts
    for _ in range(max_attempts):
        task = task_list[rng.integers(len(task_list))]
        if task["id"] not in cache:
            return task

    # Fallback: exhaustive search through all tasks
    return next((t for t in task_list if t["id"] not in cache), None)


def smart_load(raw_str: str) -> dict:
    """
    Safely load data from string using multiple parsing methods.

    Attempts to parse string as JSON first, falls back to Python literal
    evaluation if JSON decoding fails.

    Args:
        raw_str (str): String containing serialized data

    Returns:
        dict: Dictionary with parsed data

    Raises:
        ValueError: If both parsing methods fail
    """
    raw_str = raw_str.strip()
    try:
        return json.loads(raw_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw_str)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Failed to parse string: {raw_str}") from e


def remove_keys(data: dict, ex_keys: set[str]) -> dict:
    result = {}
    stack = [(data, result)]

    while stack:
        cur_dict, cur_res = stack.pop()

        for k, v in cur_dict.items():
            if k in ex_keys:
                continue

            if isinstance(v, dict):
                new_dict = {}
                cur_res[k] = new_dict
                stack.append((v, new_dict))
            else:
                cur_res[k] = v

    return result
