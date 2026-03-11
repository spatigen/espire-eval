import json
import os
from pathlib import Path

from espire_eval_common.client import EspireClient
from espire_eval_common.driver import BaseDriver
from espire_eval_common.image import encode_image

from utils.file import proj_dir, read_prompt
from utils.llm import call_qwen


class EspireDriver(BaseDriver):
    api_key: str = "QWEN_API_KEY"

    def __init__(self, log_dir: str | Path) -> None:
        super().__init__(log_dir)
        api_key = os.getenv("QWEN_API_KEY", None)
        if not api_key:
            raise RuntimeError("Environment variable QWEN_API_KEY is not set")
        else:
            self.api_key = api_key

    def localizing(
        self,
        client: EspireClient,
        config: dict,
        log_dir: Path,
        task: dict,
        obs: dict,
        **kwargs,
    ) -> tuple[int, int]:
        # Build system prompt
        if task["action"] not in ("pick", "place"):
            client.log(
                f"Failed to parse the action: {task['action']}",
                level="error",
            )
            raise KeyError

        system_prompt = read_prompt(
            proj_dir()
            / config["model"]["prompt_dir"]
            / task["action"]
            / "localizing.txt"
        )

        # Query qwen
        try:
            response = call_qwen(
                model_name=config["model"]["name"],
                api_key=self.api_key,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": task["instruction"]},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(obs['rgb_path'])}",
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ],
                temperature=config["model"]["temperature"],
                max_tokens=config["model"]["max_tokens"],
            )
        except Exception as e:
            client.log(
                f"Failed for Qwen to inference localizing:\n{e}",
                level="error",
            )
            raise RuntimeError

        client.log(f"Model Output\n{response}")

        # Parse response from qwen
        try:
            if "```json" in response:
                res = response[
                    response.find("```json") + 7 : response.rfind("```")
                ].strip()
            else:
                res = response.strip()

            try:
                point_2d = json.loads(res)["point_2d"]
            except Exception as e:
                client.log(f"Parse JSON failed for:\n{e}", level="WARNING")
                start_pt = response.find("[")
                end_pt = response.find("]", start_pt)
                point_2d = eval(response[start_pt : end_pt + 1])

            target_x, target_y = point_2d
            point = (
                int(round(target_y / 1000 * obs["shape"]["row"])),
                int(round(target_x / 1000 * obs["shape"]["col"])),
            )
        except Exception as e:
            client.log(f"Failed to parse Qwen's output for:\n{e}")
            raise SyntaxError

        return point

    def moving(
        self,
        client: EspireClient,
        config: dict,
        log_dir: Path,
        task: dict,
        obs: dict,
        **kwargs,
    ) -> tuple[int, int]:
        # Build system prompt
        if task["action"] == "pick":
            system_prompt = read_prompt(
                proj_dir() / config["model"]["prompt_dir"] / "pick" / "moving_1.txt",
            )
            move_instruction = f"Reach for the book that needs to be grabbed; this book is marked by a bounding box of the {obs['bbox_color']} color."
        elif task["action"] == "place":
            system_prompt = read_prompt(
                proj_dir() / config["model"]["prompt_dir"] / "place" / "moving_1.txt",
            )
            move_instruction = f"Reach for the location where the book in your hand needs to be placed. The placement area is marked by a bounding box in the {obs['bbox_color']} color."
        else:
            client.log(
                f"Failed to parse the action: {task['action']}",
                level="error",
            )
            raise KeyError

        # Query qwen
        try:
            response = call_qwen(
                model_name=config["model"]["name"],
                api_key=self.api_key,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": move_instruction,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(obs['rgb_path'])}",
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ],
                temperature=config["model"]["temperature"],
                max_tokens=config["model"]["max_tokens"],
            )
        except Exception as e:
            client.log(
                f"Failed for Qwen to inference moving:\n{e}",
                level="error",
            )
            raise RuntimeError

        client.log(f"Model Output\n{response}")

        # Parse response from qwen
        try:
            if "```json" in response:
                res = response[
                    response.find("```json") + 7 : response.rfind("```")
                ].strip()
            else:
                res = response.strip()

            try:
                point_2d = json.loads(res)["point_2d"]
            except Exception as e:
                client.log(f"Parse JSON failed for:\n{e}", level="WARNING")
                start_pt = response.find("[")
                end_pt = response.find("]", start_pt)
                point_2d = eval(response[start_pt : end_pt + 1])

            target_x, target_y = point_2d
            row = obs["shape"]["row"]
            col = obs["shape"]["col"]
            point = (
                int(round(target_y / 1000 * row)),
                int(round(target_x / 1000 * col)),
            )
        except Exception as e:
            client.log(f"Failed to parse Qwen's output for:\n{e}")
            raise SyntaxError

        return point
