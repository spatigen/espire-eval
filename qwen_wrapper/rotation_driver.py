import json
import os
from pathlib import Path

from espire_eval_common.client import EspireClient
from espire_eval_common.driver import BaseDriver
from espire_eval_common.image import draw_points_on_image, encode_image

from utils.file import proj_dir, read_prompt
from utils.llm import call_qwen


class RotEspireDriver(BaseDriver):
    api_key: str = "QWEN_API_KEY"
    rotation: tuple[float | None, float | None, float | None]

    def __init__(self, log_dir: str | Path) -> None:
        super().__init__(log_dir)
        api_key = os.getenv("QWEN_API_KEY", None)
        if not api_key:
            raise RuntimeError("Environment variable QWEN_API_KEY is not set")
        else:
            self.api_key = api_key

        self.rotation_require = ("pitch", None, "roll")
        self.rotation = (0.0, None, 0.0)

    def localizing(
        self,
        client: EspireClient,
        config: dict,
        log_dir: Path,
        task: dict,
        obs: dict,
        **kwargs,
    ) -> tuple[int, int]:
        return 0, 0  # nonsense for server requirement

    def do_moving(
        self, client: EspireClient, config: dict, log_dir: Path, task: dict
    ) -> None:
        # Get observation for moving
        observation = self.get_obs_for_first_moving(client)

        # Begin loop
        moving_trials = config.get("moving_trials", 5)
        for trial in range(moving_trials):
            client.log(f"Moving trial {trial + 1} / {moving_trials}")

            # Moving
            point = self.moving(client, config, log_dir, task, observation)
            rotation = self.query_rotation(
                client, config, log_dir, task, observation, point, trial
            )
            self.rotation = (
                rotation[0] if self.rotation_require[0] is not None else None,
                rotation[1] if self.rotation_require[1] is not None else None,
                rotation[2] if self.rotation_require[2] is not None else None,
            )
            moving_result = self.execute_moving(client, observation, point, trial)

            # Stop according to the result
            if moving_result["goal_met"]:
                self.get_obs_for_moving(client)
                break

            # Prepare for the next moving
            observation = self.get_obs_for_moving(client)

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
                res = response[response.find("```json") + 7 : response.rfind("```")]
            else:
                res = response
            point_2d = json.loads(res)["point_2d"]
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

    def query_rotation(
        self,
        client: EspireClient,
        config: dict,
        log_dir: Path,
        task: dict,
        obs: dict,
        point: tuple[int, int],
        trial: int,
    ) -> tuple[float, float, float]:
        # Create visualization of moving point
        rgb_path = Path(obs["rgb_path"])
        new_name = f"{trial:02d}_moving_{rgb_path.name}"
        output_path = rgb_path.with_name(new_name)
        draw_points_on_image(obs["rgb_path"], point[0], point[1], output_path)

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
            / "query_rotation.txt"
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
                            {
                                "type": "text",
                                "text": f"{task['instruction']}\nThe rotation delta(s) you should predict: {self.rotation_require}.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(output_path)}",
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
                f"Failed for Qwen to inference rotation:\n{e}",
                level="error",
            )
            raise RuntimeError

        client.log(f"Model Output\n{response}")

        # Parse response from qwen
        try:
            if "```json" in response:
                res = response[response.find("```json") + 7 : response.rfind("```")]
            else:
                res = response
            pred_rotation = json.loads(res)["rotation"]
            pred_rotation = [x if x is not None else 0.0 for x in pred_rotation]
        except Exception as e:
            client.log(f"Failed to parse Qwen's output for:\n{e}")
            raise SyntaxError

        return (
            float(pred_rotation[0]),
            float(pred_rotation[1]),
            float(pred_rotation[2]),
        )

    def execute_moving(
        self, client: EspireClient, obs: dict, point: tuple[int, int], trial: int
    ) -> dict:
        rgb_path = Path(obs["rgb_path"])
        new_name = f"{trial:02d}_moving_{rgb_path.name}"
        output_path = rgb_path.with_name(new_name)

        try:
            # Execute moving on server
            move_res = client.move_to(
                point,
                rotation_delta=self.rotation,
            )["content"]
            move_res["moving_img_path"] = str(output_path.resolve())
        except Exception as e:
            client.log(f"Failed to move from server for:\n{e}", level="ERROR")
            raise ConnectionError

        return move_res
