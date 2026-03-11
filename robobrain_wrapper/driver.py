from pathlib import Path

from espire_eval_common.client import EspireClient
from espire_eval_common.driver import BaseDriver
from sentence_transformers import SentenceTransformer

from robobrain_wrapper.inference import UnifiedInference


class EspireDriver(BaseDriver):
    model: UnifiedInference
    sentence_transformer: SentenceTransformer

    def __init__(self, log_dir: str | Path, model_path: str = "") -> None:
        super().__init__(log_dir)
        if model_path == "":
            self.model = UnifiedInference("BAAI/RoboBrain2.0-7B")
        else:
            self.model = UnifiedInference(model_path)

        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")

    def localizing(
        self,
        client: EspireClient,
        config: dict,
        log_dir: Path,
        task: dict,
        obs: dict,
        **kwargs,
    ) -> tuple[int, int]:
        # Query robobrain2
        try:
            response = self.model.inference(
                task["instruction"],
                obs["rgb_path"],
                task="pointing",
                plot=True,
                enable_thinking=True,
                do_sample=True,
                img_dir=Path(log_dir / "rgb"),
            )
        except Exception as e:
            client.log(
                f"Failed for RoboBrain2 to inference localizing:\n{e}",
                level="error",
            )
            raise RuntimeError

        client.log(f"Model Output\n{response}")

        # Parse response from robobrain2
        try:
            point = eval(response["answer"])[0]
            point = (int(point[1]), int(point[0]))  # row, col
        except Exception as e:
            client.log(f"Failed to parse RoboBrain2's output for:\n{e}")
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
        if task["action"] == "pick":
            move_instruction = f"Reach for the book that needs to be grabbed; this book is marked by a bounding box of the {obs['bbox_color']} color."
        elif task["action"] == "place":
            move_instruction = f"Reach for the location where the book in your hand needs to be placed. The placement area is marked by a bounding box in the {obs['bbox_color']} color."
        else:
            client.log(
                f"Failed to parse the action: {task['action']}",
                level="error",
            )
            raise KeyError

        # Query robobrain2
        try:
            response = self.model.inference(
                move_instruction,
                obs["rgb_path"],
                task="trajectory",
                plot=True,
                enable_thinking=True,
                do_sample=True,
                img_dir=Path(log_dir / "rgb"),
            )
        except Exception as e:
            client.log(
                f"Failed for RoboBrain2 to inference moving:\n{e}",
                level="error",
            )
            raise RuntimeError

        client.log(f"Model Output\n{response}")

        # Parse response from robobrain2
        try:
            point = eval(response["answer"])[-1]
            point = (int(point[1]), int(point[0]))  # row, col
        except Exception as e:
            client.log(f"Failed to parse RoboBrain2's output for:\n{e}")
            raise SyntaxError

        return point


class LocalEspireDriver(EspireDriver):
    model: UnifiedInference
    sentence_transformer: SentenceTransformer

    def __init__(self, log_dir: str | Path) -> None:
        super().__init__(log_dir, "RoboBrain2.0-7B")
