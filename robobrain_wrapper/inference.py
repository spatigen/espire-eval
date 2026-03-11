import os
import re
from typing import Union

import cv2
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class UnifiedInference:
    """
    A unified class for performing inference using RoboBrain models.
    Supports both 3B (non-thinking) and 7B/32B (thinking) models.

    # Example:
    ```python
    model = UnifiedInference("BAAI/RoboBrain2.0-7B")
    prompt = "Your goal is to grasp the target book from its spine side.\nThe target book is masked in magenta."
    image = "img.png"

    pred = model.inference(
        prompt, image, task="trajectory", plot=True, enable_thinking=True, do_sample=True
    )
    print(f"Prediction:\n{pred}")
    ```

    # Expected output:
    ```text
    Prediction: (as an example)
    {
        'thinking': 'From the visual input, there is a clear division between two mugs with distinct colors: one blue and one green. The blue mug is positioned towards the left side, while the green mug is on the right side of the scene. The task requires identifying spots between these two mugs.\n\nMy visual processing allows me to focus on the area between the blue mug and the green mug. This region appears to be moderately uniform in texture relative to the surrounding surfaces. I see no significant objects or textures interfering directly between them, suggesting an open space suitable for placing points.\n\nTo fulfill the requirement of identifying multiple spots within this vacant region, I will select distinct pixel coordinates that lie between the mugs, ensuring they are not centered on the mugs themselves or any visible obstructions. My approach involves choosing several points distributed across this gap, maintaining a reasonable spread to reflect "several" distinct locations.\n\nVerification ensures each selected point lies strictly within the visible vacant space between the two cups, away from the edges or any mugs\' contours. Distinctness among points is confirmed to ensure no overlap occurs.\n\nThus, based on direct visual analysis and task requirements, identified points within the vacant area between the two mugs include (376, 309), (332, 357), (384, 335), (343, 296), (348, 327), (328, 346), (371, 322), (392, 303), (380, 324), and (337, 295). These points satisfy all conditions specified by the task.',
        'answer': '[(376, 309), (332, 357), (384, 335), (343, 296), (348, 327), (328, 346), (371, 322), (392, 303), (380, 324), (337, 295)]'
    }
    ```
    """

    def __init__(self, model_id="BAAI/RoboBrain2.0-7B", device_map="auto"):
        """
        Initialize the model and processor.

        Args:
            model_id (str): Path or Hugging Face model identifier
            device_map (str): Device mapping strategy ("auto", "cuda:0", etc.)
        """
        print("Loading Checkpoint ...")
        self.model_id = model_id
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.supports_thinking = self._check_thinking_support(model_id)
        print(f"Model thinking support: {self.supports_thinking}")

    def _check_thinking_support(self, model_id):
        """Check if the model supports thinking mode based on model identifier."""
        model_name = model_id.lower()
        if "3b" in model_name:
            return False
        elif any(size in model_name for size in ["7b", "32b"]):
            return True
        else:
            return True

    def inference(
        self,
        text: str,
        image: Union[list, str],
        task="general",
        plot=False,
        enable_thinking=None,
        do_sample=True,
        temperature=0.7,
        img_dir=None,
    ):
        """
        Perform inference with text and images input.

        Args:
            text (str): The input text prompt.
            image (Union[list,str]): The input image(s) as a list of file paths or a single file path.
            task (str): The task type, e.g., "general", "pointing", "affordance", "trajectory", "grounding".
            plot (bool): Whether to plot results on image.
            enable_thinking (bool, optional): Whether to enable thinking mode.
                                            If None, auto-determined based on model capability.
            do_sample (bool): Whether to use sampling during generation.
            temperature (float): Temperature for sampling.
            img_dir (Path): Path of image directory
        """

        if isinstance(image, str):
            image = [image]

        assert task in [
            "general",
            "pointing",
            "affordance",
            "trajectory",
            "grounding",
        ], (
            f"Invalid task type: {task}. Supported tasks are 'general', 'pointing', 'affordance', 'trajectory', 'grounding'."
        )
        assert task == "general" or (
            task in ["pointing", "affordance", "trajectory", "grounding"]
            and len(image) == 1
        ), (
            "Pointing, affordance, grounding, and trajectory tasks require exactly one image."
        )

        if enable_thinking is None:
            enable_thinking = self.supports_thinking
        elif enable_thinking and not self.supports_thinking:
            print(
                "Warning: Thinking mode requested but not supported by this model. Disabling thinking."
            )
            enable_thinking = False

        if task == "pointing":
            print("Pointing task detected. Adding pointing prompt.")
            text = f"{text}. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should indicate the normalized pixel locations of the points in the image."
        elif task == "affordance":
            print("Affordance task detected. Adding affordance prompt.")
            text = f'You are a robot using the joint control. The task is "{text}". Please predict a possible affordance area of the end effector.'
        elif task == "trajectory":
            print("Trajectory task detected. Adding trajectory prompt.")
            text = f'You are a robot using the joint control. The task is "{text}". Please predict up to 10 key trajectory points to complete the task. Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x and y coordinates of a point.'
        elif task == "grounding":
            print("Grounding task detected. Adding grounding prompt.")
            text = f"Please provide the bounding box coordinate of the region this sentence describes: {text}."

        print(f"\n{'=' * 20} INPUT {'=' * 20}\n{text}\n{'=' * 47}\n")

        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {
                            "type": "image",
                            "image": path
                            if path.startswith("http")
                            else f"file://{path}",
                        }
                        for path in image
                    ],
                    {"type": "text", "text": f"{text}"},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if enable_thinking:
            print("Thinking enabled.")
            text = f"{text}<think>"
        elif self.supports_thinking:
            print("Thinking disabled (but supported).")
            text = f"{text}<think></think><answer>"

        image_inputs, video_inputs = process_vision_info(messages)  # type: ignore
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        print("Running inference ...")
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=768, do_sample=do_sample, temperature=temperature
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if enable_thinking and self.supports_thinking:
            raw_output = output_text[0] if output_text else ""
            if "</think>" in raw_output:
                thinking_text = (
                    raw_output.split("</think>")[0].replace("<think>", "").strip()
                )
                answer_text = (
                    raw_output.split("</think>")[1]
                    .replace("<answer>", "")
                    .replace("</answer>", "")
                    .strip()
                )
            else:
                thinking_text = raw_output.replace("<think>", "").strip()
                answer_text = ""
        elif self.supports_thinking:
            # Thinking disabled but supported
            raw_output = output_text[0] if output_text else ""
            thinking_text = ""
            answer_text = (
                raw_output.replace("<answer>", "").replace("</answer>", "").strip()
            )
        else:
            # No thinking support (3B models)
            raw_output = output_text[0] if output_text else ""
            thinking_text = ""
            answer_text = raw_output

        # print(f"Raw output: {output_text}")
        # if thinking_text:
        #     print(f"Thinking: {thinking_text}")
        # print(f"Answer: {answer_text}")

        # Plotting functionality
        if plot and task in ["pointing", "affordance", "trajectory", "grounding"]:
            print("Plotting enabled. Drawing results on the image ...")

            plot_points, plot_boxes, plot_trajectories = None, None, None
            result_text = answer_text  # Use the processed answer text for plotting

            image_name_to_save = None
            if task == "trajectory":
                trajectory_pattern = r"(\d+),\s*(\d+)"
                trajectory_points = re.findall(trajectory_pattern, result_text)
                plot_trajectories = [[(int(x), int(y)) for x, y in trajectory_points]]
                print(f"Extracted trajectory points: {plot_trajectories}")
                image_name_to_save = os.path.basename(image[0]).replace(
                    ".", "_with_trajectory_annotated."
                )
            elif task == "pointing":
                point_pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"
                points = re.findall(point_pattern, result_text)
                plot_points = [(int(x), int(y)) for x, y in points]
                print(f"Extracted points: {plot_points}")
                image_name_to_save = os.path.basename(image[0]).replace(
                    ".", "_with_pointing_annotated."
                )
            elif task == "affordance":
                box_pattern = r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
                boxes = re.findall(box_pattern, result_text)
                plot_boxes = [
                    [int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in boxes
                ]
                print(f"Extracted bounding boxes: {plot_boxes}")
                image_name_to_save = os.path.basename(image[0]).replace(
                    ".", "_with_affordance_annotated."
                )
            elif task == "grounding":
                box_pattern = r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
                boxes = re.findall(box_pattern, result_text)
                plot_boxes = [
                    [int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in boxes
                ]
                print(f"Extracted bounding boxes: {plot_boxes}")
                image_name_to_save = os.path.basename(image[0]).replace(
                    ".", "_with_grounding_annotated."
                )

            assert image_name_to_save is not None, "Image name to save is None"

            if img_dir is None:
                os.makedirs("result", exist_ok=True)
                image_path_to_save = os.path.join("result", image_name_to_save)
            else:
                image_path_to_save = img_dir / image_name_to_save

            self.draw_on_image(
                image[0],
                points=plot_points,
                boxes=plot_boxes,
                trajectories=plot_trajectories,
                output_path=image_path_to_save,
            )

        # Return unified format
        result = {"answer": answer_text}
        if thinking_text:
            result["thinking"] = thinking_text

        return result

    def draw_on_image(
        self, image_path, points=None, boxes=None, trajectories=None, output_path=None
    ):
        """
        Draw points, bounding boxes, and trajectories on an image

        Parameters:
            image_path: Path to the input image
            points: List of points in format [(x1, y1), (x2, y2), ...]
            boxes: List of boxes in format [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
            trajectories: List of trajectories in format [[(x1, y1), (x2, y2), ...], [...]]
            output_path: Path to save the output image. Default adds "_annotated" suffix to input path
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Unable to read image: {image_path}")

            # Draw points
            if points:
                for point in points:
                    x, y = point
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red solid circle

            # Draw bounding boxes
            if boxes:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(
                        image, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )  # Green box, line width 2

            # Draw trajectories
            if trajectories:
                for trajectory in trajectories:
                    if len(trajectory) < 2:
                        continue  # Need at least 2 points to form a trajectory
                    # Connect trajectory points with lines
                    for i in range(1, len(trajectory)):
                        cv2.line(
                            image, trajectory[i - 1], trajectory[i], (255, 0, 0), 2
                        )  # Blue line, width 2
                    # Draw a larger point at the trajectory end
                    end_x, end_y = trajectory[-1]
                    cv2.circle(
                        image, (end_x, end_y), 7, (255, 0, 0), -1
                    )  # Blue solid circle, slightly larger

            # Determine output path
            if not output_path:
                name, ext = os.path.splitext(image_path)
                output_path = f"{name}_annotated{ext}"

            # Save the result
            cv2.imwrite(output_path, image)
            print(f"Annotated image saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error processing image: {e}")
            return None
