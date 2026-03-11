import json
import os
from pathlib import Path

from espire_eval_common.client import EspireClient
from espire_eval_common.driver import ReflectionDriver, TwoViewDriver
from espire_eval_common.image import encode_image

from utils.file import proj_dir, read_prompt
from utils.llm import call_qwen


class RTEspireDriver(ReflectionDriver, TwoViewDriver):
    api_key: str = "QWEN_API_KEY"
    reflection_prompt: str
    reflection_message: str

    def __init__(self, log_dir: str | Path) -> None:
        super().__init__(log_dir)
        api_key = os.getenv("QWEN_API_KEY", None)
        if not api_key:
            raise RuntimeError("Environment variable QWEN_API_KEY is not set")
        else:
            self.api_key = api_key

        self.reflection_prompt = 'The previous visual {0} attempt for the following instruction was incorrect: "{1}". The red circles in the image mark the wrong {0} regions. Please carefully analyze the image and the instruction again, understand why the previous prediction was wrong, and describe what should be corrected in the next attempt.'
        self.reflection_message = "Here is a reflection on the previous failed attempt and some suggestions:\n```text{}```\nNow complete this task: "

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

        # Deal with reflection
        reflection_messages = []
        if (
            kwargs.get("reflection", None) is not None
            and kwargs.get("prev_localizing_result", None) is not None
        ):
            reflection = kwargs["reflection"]
            prev_localizing_result = kwargs["prev_localizing_result"]

            reflection_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.reflection_prompt.format(
                                "localizing", task["instruction"]
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encode_image(prev_localizing_result['localizing_img_path'])}",
                                "detail": "high",
                            },
                        },
                    ],
                },
                {"role": "assistant", "content": reflection},
            ]

        # Query qwen
        try:
            response = call_qwen(
                model_name=config["model"]["name"],
                api_key=self.api_key,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *reflection_messages,
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
                res = response[response.find("```json") + 7 : response.rfind("```")]
            else:
                res = response
            point_2d = json.loads(res)["point_2d"]
            target_x, target_y = point_2d
            point = (
                int(round(target_y / 1000 * obs["shape"]["row"])),
                int(round(target_x / 1000 * obs["shape"]["col"])),
            )
        except Exception as e:
            client.log(f"Failed to parse Qwen's output for:\n{e}")
            raise SyntaxError

        return point

    def reflect_localizing(
        self,
        client: EspireClient,
        config: dict,
        log_dir: Path,
        task: dict,
        localizing_result: dict,
    ) -> str:
        # Build system prompt
        system_prompt = read_prompt(
            proj_dir()
            / config["model"]["prompt_dir"]
            / task["action"]
            / "reflect_localizing.txt"
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
                                "text": self.reflection_prompt.format(
                                    "localizing", task["instruction"]
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(localizing_result['localizing_img_path'])}",
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
                f"Failed for Qwen to reflect localizing:\n{e}",
                level="error",
            )
            raise RuntimeError

        client.log(f"Model Output\n{response}")

        return response

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
                proj_dir() / config["model"]["prompt_dir"] / "pick" / "moving_2.txt",
            )
            move_instruction = f"Reach for the book that needs to be grabbed; this book is marked by a bounding box of the {obs['bbox_color']} color."
        elif task["action"] == "place":
            system_prompt = read_prompt(
                proj_dir() / config["model"]["prompt_dir"] / "place" / "moving_2.txt",
            )
            move_instruction = f"Reach for the location where the book in your hand needs to be placed. The placement area is marked by a bounding box in the {obs['bbox_color']} color."
        else:
            client.log(
                f"Failed to parse the action: {task['action']}",
                level="error",
            )
            raise KeyError

        # Add move_instruction for reflection
        task["move_instruction"] = move_instruction

        # Deal with reflection
        has_reflection = kwargs.get("reflection", None) is not None

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
                                "text": (
                                    self.reflection_message.format(kwargs["reflection"])
                                    if has_reflection
                                    else ""
                                )
                                + move_instruction,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(obs['world_view']['rgb_path'] if has_reflection else obs['ego_view']['rgb_path'])}",
                                    "detail": "high",
                                },
                            },
                            {  # TwoViewDriver: world view first, and then ego view
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
            row = (
                obs["ego_view"]["shape"]["row"]
                if has_reflection
                else obs["world_view"]["shape"]["row"]
            )
            col = (
                obs["ego_view"]["shape"]["col"]
                if has_reflection
                else obs["world_view"]["shape"]["col"]
            )
            point = (
                int(round(target_y / 1000 * row)),
                int(round(target_x / 1000 * col)),
            )
        except Exception as e:
            client.log(f"Failed to parse Qwen's output for:\n{e}")
            raise SyntaxError

        return point

    def reflect_moving(
        self,
        client: EspireClient,
        config: dict,
        log_dir: Path,
        task: dict,
        obs: dict,
        new_obs: dict,
        moving_result: dict,
    ) -> str:
        # Build system prompt
        if task["action"] == "pick":
            system_prompt = read_prompt(
                proj_dir()
                / config["model"]["prompt_dir"]
                / "pick"
                / "reflect_moving.txt"
            )
        elif task["action"] == "place":
            system_prompt = read_prompt(
                proj_dir()
                / config["model"]["prompt_dir"]
                / "place"
                / "reflect_moving.txt"
            )
        else:
            client.log(
                f"Failed to parse the action: {task['action']}",
                level="error",
            )
            raise KeyError

        # Some info
        if moving_result["status"] == "retreat":
            some_error_info = (
                "The target point itself is unreachable, which could be due to the target being outside "
                "the robot's reachable workspace, joint limits preventing a valid configuration, "
                "singularities in the robot's kinematic chain, or numerical issues in the solver algorithm. "
                "However, the area around the target is reachable, so the robot moved to a nearby reachable point."
            )

        elif moving_result["status"] == "refocus":
            some_error_info = (
                "Not only is the target point unreachable, but the surrounding area within a certain range "
                "is also unreachable. This may be caused by the target being outside the robot's reachable "
                "workspace, joint limits preventing a valid configuration, singularities in the robot's "
                "kinematic chain, or numerical issues in the solver algorithm. The robot therefore failed "
                "to find any feasible alternative target."
            )

        elif moving_result["status"] == "success":
            some_error_info = (
                "The robot successfully moved to the previously specified target position. "
                "A valid configuration was found and executed without encountering reachability, "
                "joint limit, singularity, or solver-related issues."
            )
        else:
            some_error_info = ""

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
                                "text": self.reflection_prompt.format(
                                    "moving", task["move_instruction"]
                                )
                                + some_error_info,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(obs['world_view']['rgb_path'])}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(moving_result['moving_img_path'])}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(new_obs['world_view']['rgb_path'])}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(new_obs['ego_view']['rgb_path'])}",
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
                f"Failed for Qwen to reflect moving:\n{e}",
                level="error",
            )
            raise RuntimeError

        client.log(f"Model Output\n{response}")

        return response
