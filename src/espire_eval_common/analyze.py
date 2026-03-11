import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

from espire_eval_common.utils import smart_load


def traverse(dir_path: Path) -> Iterator[tuple[str, str]]:
    """
    Traverse subdirectories and extract log file contents.

    Iterates through all subdirectories in the given path, attempts to read
    'run.log' files from each, and yields directory names with their contents.

    Args:
        dir_path (Path): Object representing the directory to traverse

    Yields:
        Tuple containing subdirectory name and log file content
    """
    # Validate directory existence and type
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Invalid directory: {dir_path}")
        return

    # Process each subdirectory
    for subdir in dir_path.iterdir():
        if not subdir.is_dir():
            continue

        log_file = subdir / "run.log"
        if not log_file.exists():
            print(f"No run.log file found in {subdir}")
            continue

        try:
            # Read and yield non-empty log files
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if content:  # More Pythonic than len(content) != 0
                yield subdir.name, content

        except Exception as e:
            print(f"Failed to read {log_file}: {e}")


def statistics(result: dict) -> None:
    """
    Print statistical analysis of experiment results.

    Calculates and displays various metrics including sample counts,
    success rates for localizing and moving operations, and error statistics.

    Args:
        res (dict): Dictionary containing experiment results keyed by sample names
    """
    total_samples = len(result)
    print(f"Number of samples:\n  {total_samples}")

    #############################################
    # Localizing statistics
    #############################################
    # Calculate success rate with zero-division protection
    localizing_success = sum(
        1
        for v in result.values()
        if any(u.get("success", False) for u in v["localize"])
    )
    localizing_rate = localizing_success / total_samples if total_samples > 0 else 0.0
    print(
        f"Localizing success rate:\n  {localizing_success}/{total_samples} "
        f"≈ {localizing_rate:.4f}"
    )

    # Calculate average localizing times given success
    avg_localizing_nums = np.mean(
        [
            len(v["localize"])
            for v in result.values()
            if any(u.get("success", False) for u in v["localize"])
        ]
    )
    print(f"Average localizing times given success:\n  {avg_localizing_nums:.2f}")

    #############################################
    # Moving statistics
    #############################################
    # Calculate success rate with zero-division protection
    moving_success = sum(
        1 for v in result.values() if any(u.get("success", False) for u in v["move"])
    )
    moving_rate = moving_success / total_samples if total_samples > 0 else 0.0
    print(
        f"Moving success rate:\n  {moving_success}/{total_samples} ≈ {moving_rate:.4f}"
    )

    # Calculate success rate (localize && move)
    combined_success = sum(
        1
        for v in result.values()
        if any(u.get("success", False) for u in v["localize"])
        and any(u.get("success", False) for u in v["move"])
    )
    combined_rate = combined_success / total_samples if total_samples > 0 else 0.0
    print(
        f"Localizing and moving success rate:\n  {combined_success}/{total_samples} ≈ {combined_rate:.4f}"
    )

    # Calculate success rate (localize && move && finish_goal)
    combined_success = sum(
        1
        for v in result.values()
        if any(u.get("success", False) for u in v["localize"])
        and any(u.get("success", False) for u in v["move"])
        and any(u.get("execution", {}).get("finish_goal", False) for u in v["end_task"])
    )
    combined_rate = combined_success / total_samples if total_samples > 0 else 0.0
    print(
        f"Localizing and moving and finish_goal success rate:\n  {combined_success}/{total_samples} ≈ {combined_rate:.4f}"
    )

    # Calculate average moving times given success
    avg_moving_nums = np.mean(
        [
            len(v["move"])
            for v in result.values()
            if any(u.get("success", False) for u in v["move"])
        ]
    )
    print(f"Average moving times given success:\n  {avg_moving_nums:.2f}")

    # Calculate average of final distance when success
    avg_distance = np.mean(
        [
            u["distance"]
            for v in result.values()
            for u in v["move"]
            if u.get("success", False)
        ]
    )
    print(f"Average distance at success:\n  {avg_distance:.2f}")

    # Calculate average of last distance before success
    last_distance_lst = []
    for v in result.values():
        last_dist = None
        for u in v["move"]:
            if u.get("success", False):
                if last_dist is None:  # success at first
                    start_pos = np.array(v["set_task"][0]["wrapped_robot_ee_pos"])
                    end_pos = np.array(u["wrapped_robot_ee_pos"])
                    last_dist = np.linalg.norm(start_pos - end_pos)
                last_distance_lst.append(last_dist)
            last_dist = u.get("distance", None)
    assert len(last_distance_lst) == moving_success, (
        "Wrong len of last distance before success"
    )
    print(f"Average distance before success:\n  {np.mean(last_distance_lst):.2f}")

    #############################################
    # End statistics
    #############################################
    localizing_diff = []
    moving_diff = []
    env_step_nums = []
    for entry in result.values():
        localizing_timestamp = [
            datetime.strptime(i, "%Y-%m-%d at %H:%M:%S")
            for i in entry["localize_timestamp"]
        ]
        moving_timestamp = [
            datetime.strptime(i, "%Y-%m-%d at %H:%M:%S")
            for i in entry["move_timestamp"]
        ]

        if moving_timestamp != []:
            env_step_nums.append(0)
            env_step_nums[-1] += len(entry["move_timestamp"])  # add nums of get obs
            env_step_nums[-1] += len(entry["move_timestamp"]) - 1  # add nums of move
            if entry["end_task"] != []:
                env_step_nums[-1] += 1  # last move

        if localizing_timestamp == [] or moving_timestamp == []:
            continue

        localizing_timestamp.append(moving_timestamp[0])
        if entry["end_task"] != []:
            moving_timestamp.append(
                datetime.strptime(
                    entry["end_task"][0]["send_time"], "%Y-%m-%d at %H:%M:%S"
                )
            )
        else:
            moving_timestamp.append(
                datetime.strptime(entry["client_stopped"], "%Y-%m-%d at %H:%M:%S")
            )

        for dt1, dt2 in zip(localizing_timestamp[:-1], localizing_timestamp[1:]):
            localizing_diff.append(dt2 - dt1)
        for dt1, dt2 in zip(moving_timestamp[:-1], moving_timestamp[1:]):
            moving_diff.append(dt2 - dt1)
    print(
        f"Average time for localizing:\n  {sum(d.total_seconds() for d in localizing_diff) / len(localizing_diff):.2f} seconds"
    )
    print(
        f"Average time for moving:\n  {sum(d.total_seconds() for d in moving_diff) / len(moving_diff):.2f} seconds"
    )
    print(f"Average env step nums:\n  {np.mean(env_step_nums):.2f}")

    finish_goal = sum(
        any(u.get("execution", {}).get("finish_goal", False) for u in v["end_task"])
        for v in result.values()
    )
    print(f"Finished goal in the end:\n  {finish_goal}")
    print(
        f"Finished goal rate:\n  {finish_goal}/{total_samples} ≈ {finish_goal / total_samples:.4f}"
    )


def extract(dir_path: str | Path) -> dict:
    """
    Extract running log files.

    Processes log files to extract localizing and moving operation data.

    Args:
        dir_path (str | Path): String or Path object pointing to directory with log files

    Returns:
        dict: Dictionary containing log info
    """
    header_pattern = r"(\d{4}-\d{2}-\d{2} at \d{2}:\d{2}:\d{2}) \| ([A-Z]+) \| (.+)"
    result = {}

    # Ensure Path object
    dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path
    error_file_name = set()
    for file_name, file_content in traverse(dir_path):
        result[file_name] = {
            "set_task": [],
            "get_observation": [],
            "localize": [],
            "localize_timestamp": [],
            "move": [],
            "move_timestamp": [],
            "end_task": [],
        }
        send_cnt = recv_cnt = 0

        # Extract all log headers
        headers = list(re.finditer(header_pattern, file_content))

        for i in range(len(headers) - 1):
            log_time = headers[i].group(1).strip()
            log_level = headers[i].group(2).strip()
            log_content = headers[i].group(3).strip()
            segment = file_content[headers[i].end() : headers[i + 1].start()]

            if "Task selected" in log_content:
                result[file_name].update(smart_load(segment))
            elif "Localizing trial" in log_content:
                result[file_name]["localize_timestamp"].append(log_time)
            elif "Moving trial" in log_content:
                result[file_name]["move_timestamp"].append(log_time)
            elif "Client stopped normally" in log_content:
                result[file_name]["client_stopped"] = log_time

            if log_level == "ERROR":
                result[file_name]["error"] = log_content + segment

            if log_content == "Send":
                send_cnt += 1
                send_data = smart_load(segment)
                cmd = send_data.get("type", None)
                if (
                    cmd is not None
                    and isinstance(cmd, str)
                    and cmd.lower() in result[file_name]
                ):
                    result[file_name][cmd.lower()].append(
                        {"uid": send_data["uid"], "send_time": log_time}
                    )
                    if cmd.lower() == "get_observation":
                        result[file_name]["get_observation"][-1].update(
                            {
                                "view_name": send_data["content"]["view"],
                                "with_goal_mask": send_data["content"][
                                    "highlight_goal"
                                ],
                            }
                        )

            elif log_content == "Recv":
                recv_cnt += 1
                set_task_lst = result[file_name]["set_task"]
                get_obs_lst = result[file_name]["get_observation"]
                localizing_lst = result[file_name]["localize"]
                moving_lst = result[file_name]["move"]
                end_task_lst = result[file_name]["end_task"]

                if (
                    (not set_task_lst or "uid" not in set_task_lst[-1])
                    and (not get_obs_lst or "uid" not in get_obs_lst[-1])
                    and (not localizing_lst or "uid" not in localizing_lst[-1])
                    and (not moving_lst or "uid" not in moving_lst[-1])
                    and (not end_task_lst or "uid" not in end_task_lst[-1])
                ):
                    continue

                recv_data = smart_load(segment)
                if set_task_lst and recv_data["uid"] == set_task_lst[-1].get(
                    "uid", None
                ):
                    set_task_lst[-1].update(
                        {
                            "robot_js": recv_data["response"]["content"]["start_state"][
                                "env_state"
                            ]["robot_js"],
                            "wrapped_robot_ee_pos": recv_data["response"]["content"][
                                "start_state"
                            ]["env_state"]["wrapped_robot_ee_pos"],
                            "wrapped_robot_ee_ori": recv_data["response"]["content"][
                                "start_state"
                            ]["env_state"]["wrapped_robot_ee_ori"],
                            "world_info": recv_data["response"]["content"][
                                "start_state"
                            ]["env_state"]["world_info"],
                            "recv_time": log_time,
                        }
                    )
                elif get_obs_lst and recv_data["uid"] == get_obs_lst[-1].get(
                    "uid", None
                ):
                    get_obs_lst[-1].update({"recv_time": log_time})
                elif localizing_lst and recv_data["uid"] == localizing_lst[-1].get(
                    "uid", None
                ):
                    localizing_lst[-1].update(
                        {
                            "success": recv_data["response"]["content"]["goal_met"],
                            "distance": recv_data["response"]["content"]["post_state"][
                                "localization"
                            ]["details"]["click_distance_px"],
                            "recv_time": log_time,
                        }
                    )
                elif moving_lst and recv_data["uid"] == moving_lst[-1].get("uid", None):
                    moving_lst[-1].update(
                        {
                            "success": recv_data["response"]["content"]["goal_met"],
                            "distance": recv_data["response"]["content"]["post_state"][
                                "execution"
                            ]["details"]["distance"],
                            "wrapped_robot_ee_pos": recv_data["response"]["content"][
                                "post_state"
                            ]["execution"]["details"]["wrapped_robot_ee_pos"],
                            "wrapped_robot_ee_ori": recv_data["response"]["content"][
                                "post_state"
                            ]["execution"]["details"]["wrapped_robot_ee_ori"],
                            "recv_time": log_time,
                            "robot_js": recv_data["response"]["content"]["post_state"][
                                "env_state"
                            ]["robot_js"],
                            "world_info": recv_data["response"]["content"][
                                "post_state"
                            ]["env_state"]["world_info"],
                        }
                    )

                    if file_name not in error_file_name and not isinstance(
                        moving_lst[-1]["distance"], (int, float)
                    ):
                        print(
                            f"[Warning] File: {file_name} dit not get `distance` attribute correctly. It is recommended to manually check it. Subsequent processing has excluded this file."
                        )
                        error_file_name.add(file_name)

                elif end_task_lst and recv_data["uid"] == end_task_lst[-1].get(
                    "uid", None
                ):
                    end_task_lst[-1].update(
                        recv_data["response"]["content"]["post_state"]
                    )
                    end_task_lst[-1]["recv_time"] = log_time

        result[file_name]["send_cnt"] = send_cnt
        result[file_name]["recv_cnt"] = recv_cnt

        if "client_stopped" not in result[file_name]:
            print(
                f"[Warning] File: {file_name} did not exit normally. It is recommended to manually check it. Subsequent processing has excluded this file."
            )
            error_file_name.add(file_name)

    for file_name in error_file_name:
        result.pop(file_name)

    return result


def analyze(dir_path: str | Path) -> None:
    """
    Analyze run log files and generate performance statistics.

    Processes log files to extract localizing and moving operation data,
    then displays comprehensive statistics for all samples and by category.

    Args:
        dir_path (str | Path): String or Path object pointing to directory with log files
    """
    result = extract(dir_path)

    # Display results
    print("*" * 42)

    # Category-based analysis
    for category in ("pick", "place"):
        category_data = {k: v for k, v in result.items() if v.get("action") == category}
        print(f"{category.capitalize()}:")
        statistics(category_data)
        print("*" * 42)
