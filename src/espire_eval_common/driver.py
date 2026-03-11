from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, final

import numpy as np

from espire_eval_common.client import EspireClient
from espire_eval_common.image import draw_points_on_image
from espire_eval_common.utils import populate_task_cache, sample_new_task


class BaseDriver(ABC):
    """
    Abstract base driver for Espire

    Provides core functionality for task selection, observation handling,
    and moving coordination with automatic caching and error recovery.

    Attributes:
        category_pool (set[str]): Set of completed task UIDs for caching
        rotation_require (tuple[str | None, str | None, str | None]): Rotation(s) required for the task (default: Nones)
        seed (int): Integer seed (default: 42)
        rng (np.random.Generator): Numpy Random Number Generator
        MAX_SAMPLE_ATTEMPTS (int): Maximum attempts for task sampling (default: 1000)
    """

    category_pool: set[str]
    rotation_require: tuple[str | None, str | None, str | None]
    seed: int = 42
    rng: np.random.Generator

    MAX_SAMPLE_ATTEMPTS: int = 1000

    def __init__(self, log_dir: str | Path) -> None:
        """
        Initialize driver with task cache from existing log directory.

        Args:
            log_dir (str | Path): Directory path containing previous execution logs
        """
        self.category_pool = set(populate_task_cache(log_dir))
        self.rotation_require = (None, None, None)
        self.rng = np.random.default_rng(self.seed)

    @final
    def choose_task(self, client: EspireClient) -> dict:
        """
        Select a task from available options, avoiding previously completed ones.

        Implements intelligent sampling with fallback mechanisms for task selection.

        Args:
            client (EspireClient): Client instance for server communication

        Returns:
            dict: Selected task configuration dictionary

        Raises:
            ConnectionError: If getting task list on server fails
            RuntimeError: If no new task available
        """
        # Retrieve available task list from server
        try:
            task_list = client.get_task_list()["content"]["tasks"]
        except Exception as e:
            client.log(f"Failed to get task list from server for:\n{e}", level="ERROR")
            raise ConnectionError

        # Attempt to sample a new task
        chosen_task = sample_new_task(
            task_list, self.rng, self.category_pool, self.MAX_SAMPLE_ATTEMPTS
        )
        if chosen_task is None:
            client.log(
                "No valid task available after exhaustive searching", level="ERROR"
            )
            raise RuntimeError

        # Cache selected task and log selection
        self.category_pool.add(chosen_task["id"])
        client.log(
            "Task selected",
            {
                "id": chosen_task["id"],
                "action": chosen_task["action"],
                "instruction": chosen_task["instruction"],
            },
        )
        return chosen_task

    @final
    def set_task(self, client: EspireClient, task_uid: str) -> None:
        """
        Configure selected task on the server with rotation requirements.

        Args:
            client (EspireClient): Client instance for server communication
            task_uid (str): Unique identifier of the task to set

        Raises:
            ConnectionError: If setting task on server fails
        """
        try:
            client.set_task(task_uid, self.rotation_require)
        except Exception as e:
            client.log(f"Failed to set task from server for:\n{e}", level="ERROR")
            raise ConnectionError

    def do_localizing(
        self, client: EspireClient, config: dict, log_dir: Path, task: dict
    ) -> None:
        """
        Execute localizing phase with multiple trial attempts.

        Performs iterative localizing operations with configurable trial limits
        and automatic termination upon successful localizing evaluation.

        Args:
            client (EspireClient): Client instance for server communication
            config (dict): Configuration parameters including trial limits
            log_dir (Path): Directory for saving localizing logs and images
            task (dict): Current task configuration dictionary
        """
        # Get observation for localizing
        observation = self.get_obs_for_localizing(client)

        # Begin loop
        localizing_result = None
        localizing_trials = config.get("localizing_trials", 3)
        for trial in range(localizing_trials):
            client.log(f"Localizing trial {trial + 1} / {localizing_trials}")

            # Localizing
            point = self.localizing(client, config, log_dir, task, observation)
            localizing_result = self.execute_localizing(
                client, observation, point, trial
            )

            # Stop according to the result
            if localizing_result["goal_met"]:
                break

        if localizing_result is not None and not localizing_result["goal_met"]:
            client.localizing((0, 0), skip=True)

    def do_moving(
        self, client: EspireClient, config: dict, log_dir: Path, task: dict
    ) -> None:
        """
        Execute moving phase with iterative attempts and condition handling.

        Manages pick/place moving sequences with automatic observation updates
        and termination based on task completion status.

        Args:
            client (EspireClient): Client instance for server communication
            config (dict): Configuration parameters including trial limits
            log_dir (Path): Directory for saving moving logs and data
            task (dict): Current task configuration dictionary
        """
        # Get observation for moving
        observation = self.get_obs_for_first_moving(client)

        # Begin loop
        moving_trials = config.get("moving_trials", 5)
        for trial in range(moving_trials):
            client.log(f"Moving trial {trial + 1} / {moving_trials}")

            # Moving
            point = self.moving(client, config, log_dir, task, observation)
            moving_result = self.execute_moving(client, observation, point, trial)

            # Stop according to the result
            if moving_result["goal_met"]:
                self.get_obs_for_moving(client)
                break

            # Prepare for the next moving
            observation = self.get_obs_for_moving(client)

    @final
    def finalize_task(self, client: EspireClient) -> None:
        """
        Perform final task cleanup including observation and termination.

        Args:
            client (EspireClient): Client instance for server communication

        Raises:
            ConnectionError: If final observation or task termination fails
        """
        try:
            client.get_observation(view_name="world", highlight_goal=True)["content"]
        except Exception as e:
            client.log(
                f"Failed to get the last observation (before end_task) from server for:\n{e}",
                level="ERROR",
            )
            raise ConnectionError

        try:
            client.end_task()
        except Exception as e:
            client.log(f"Failed to end task from server for:\n{e}", level="ERROR")
            raise ConnectionError

        try:
            client.get_observation(view_name="world", highlight_goal=True)["content"]
        except Exception as e:
            client.log(
                f"Failed to get the last observation (after end_task) from server for:\n{e}",
                level="ERROR",
            )
            raise ConnectionError

    @final
    def run(
        self,
        config: dict,
        log_dir: Path,
        test_localizing: bool = True,
        test_moving: bool = True,
    ) -> None:
        """
        Execute task pipeline and manage client resource

        Args:
            config (dict): Configuration parameters for execution
            log_dir (Path): Directory for saving execution logs and data
            test_localizing (bool): If True, test localizing (True by default)
            test_moving (bool): If True, test moving (True by default)
        """
        with EspireClient(config, log_dir) as client:
            client.log(f"Task cache contains {len(self.category_pool)} entries")
            client.log(f"Using random seed: {self.seed}")

            try:
                #############################################
                # Step 1: Choose task
                #############################################
                chosen_task = self.choose_task(client)

                #############################################
                # Step 2: Set task
                #############################################
                self.set_task(client, chosen_task["id"])

                #############################################
                # Step 3: Localizing
                #############################################
                if test_localizing:
                    self.do_localizing(client, config, log_dir, chosen_task)
                else:
                    client.localizing((0, 0), skip=True)
                    client.log("Skip localizing", level="WARNING")

                #############################################
                # Step 4: Moving
                #############################################
                if test_moving:
                    self.do_moving(client, config, log_dir, chosen_task)
                else:
                    client.move_to((0, 0), skip=True)
                    client.log("Skip moving", level="WARNING")

                #############################################
                # Step 5: End task
                #############################################
                self.finalize_task(client)
            except ConnectionError:
                client.log("Connection Error", level="ERROR")
                return
            except Exception:
                return

    def get_obs_for_localizing(self, client: EspireClient) -> dict:
        """
        Retrieve observation data for localizing phase from world view.

        Args:
            client (EspireClient): Client instance for server communication

        Returns:
            dict: Observation data containing images and calibration

        Raises:
            ConnectionError: If getting observation on server fails
        """
        try:
            return client.get_observation("world", highlight_goal=False)["content"]
        except Exception as e:
            client.log(
                f"Failed to get observation from server for:\n{e}", level="ERROR"
            )
            raise ConnectionError

    @abstractmethod
    def localizing(
        self,
        client: EspireClient,
        config: dict,
        log_dir: Path,
        task: dict,
        obs: dict,
        **kwargs,
    ) -> tuple[int, int]:
        """
        Perform localizing operation to determine target point coordinates.

        Args:
            client (EspireClient): Client instance for server communication
            config (dict): Configuration parameters
            log_dir (Path): Directory for saving log files and images
            task (dict): Current task configuration
            obs (dict): Observation data from current world view
            **kwargs: Additional keyword arguments for implementation-specific options

        Returns:
            tuple[int, int]: (row, column) point coordinates for localizing

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        raise NotImplementedError("Localizing is not implemented!")

    @final
    def execute_localizing(
        self, client: EspireClient, obs: dict, point: tuple[int, int], trial: int
    ) -> dict:
        """
        Execute localizing operation with visualization.

        Args:
            client (EspireClient): Client instance
            obs (dict): World view observation for localizing
            point (tuple[int, int]): Target point coordinates with (row, col)
            trial (int): Trial nums

        Returns:
            dict: Server response containing localizing evaluation results

        Raises:
            ConnectionError: If localizing request to server fails
        """
        # Create visualization of localizing point
        rgb_path = Path(obs["rgb_path"])
        new_name = f"{trial:02d}_localizing_{rgb_path.name}"
        output_path = rgb_path.with_name(new_name)
        draw_points_on_image(obs["rgb_path"], point[0], point[1], output_path)

        try:
            # Execute localizing on server
            ground_res = client.localizing(point)["content"]
            ground_res["localizing_img_path"] = str(output_path.resolve())
        except Exception as e:
            client.log(f"Failed to localizing from server for:\n{e}", level="ERROR")
            raise ConnectionError

        return ground_res

    def get_obs_for_first_moving(self, client: EspireClient) -> dict:
        """
        Retrieve observation data for first moving phase with goal mask.

        Args:
            client (EspireClient): Client instance for server communication

        Returns:
            dict: Observation data with goal mask enabled

        Raises:
            ConnectionError: If getting observation on server fails
        """
        try:
            return client.get_observation("world", highlight_goal=True)["content"]
        except Exception as e:
            client.log(
                f"Failed to get observation from server for:\n{e}", level="ERROR"
            )
            raise ConnectionError

    @abstractmethod
    def moving(
        self,
        client: EspireClient,
        config: dict,
        log_dir: Path,
        task: dict,
        obs: dict,
        **kwargs,
    ) -> tuple[int, int]:
        """
        Execute moving command to target position.

        Args:
            client (EspireClient): Client instance for server communication
            config (dict): Configuration parameters
            log_dir (Path): Directory for saving log files
            task (dict): Current task configuration
            obs (dict): Observation data from current view
            **kwargs: Additional keyword arguments for implementation-specific options

        Returns:
            tuple[int, int]: (row, column) point coordinates for moving

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        raise NotImplementedError("Moving is not implemented!")

    def execute_moving(
        self, client: EspireClient, obs: dict, point: tuple[int, int], trial: int
    ) -> dict:
        """
        Execute moving operation to target point with visualization.

        Args:
            client (EspireClient): Client instance for server communication
            obs (dict): Observation data from current view
            point (tuple[int, int]): Target point coordinates (row, col) for moving
            trial (int): Trial nums

        Returns:
            dict: Server response containing moving execution results

        Raises:
            ConnectionError: If moving request to server fails
        """
        # Create visualization of moving point
        rgb_path = Path(obs["rgb_path"])
        new_name = f"{trial:02d}_moving_{rgb_path.name}"
        output_path = rgb_path.with_name(new_name)
        draw_points_on_image(obs["rgb_path"], point[0], point[1], output_path)

        try:
            # Execute moving on server
            move_res = client.move_to(point)["content"]
            move_res["moving_img_path"] = str(output_path.resolve())
        except Exception as e:
            client.log(f"Failed to move from server for:\n{e}", level="ERROR")
            raise ConnectionError

        return move_res

    def get_obs_for_moving(self, client: EspireClient) -> dict:
        """
        Retrieve ego-view observation data for second moving phase.

        Args:
            client (EspireClient): Client instance for server communication

        Returns:
            dict: Ego-view observation data with goal mask

        Raises:
            ConnectionError: If getting observation on server fails
        """
        try:
            return client.get_observation("ego", highlight_goal=True)["content"]
        except Exception as e:
            client.log(
                f"Failed to get observation from server for:\n{e}", level="ERROR"
            )
            raise ConnectionError


class ReflectionDriver(BaseDriver):
    """
    Localizing/Moving with Reflection

    Extends base driver with iterative reflection capabilities for error correction
    and performance optimization through trial analysis and adaptive strategy adjustment.

    Implements reflective learning cycles that analyze previous attempt outcomes
    to improve subsequent localizing and moving decisions.
    """

    @abstractmethod
    def reflect_localizing(
        self,
        client: EspireClient,
        config: dict,
        log_dir: Path,
        task: dict,
        localizing_result: dict,
    ) -> str:
        """
        Analyze localizing results and generate reflection insights for improvement.

        Processes evaluation feedback from failed localizing attempts to produce
        actionable insights that guide subsequent localizing strategy adjustments.

        Args:
            client (EspireClient): Client instance for logging and communication
            config (dict): Configuration parameters for reflection analysis
            log_dir (Path): Directory for saving reflection logs and analysis
            task (dict): Current task configuration for contextual understanding
            localizing_result (dict): Server response containing localizing evaluation metrics

        Returns:
            str: Reflection insights as natural language text for next attempt guidance

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        raise NotImplementedError("Reflect localizing is not implemented!")

    @final
    def do_localizing(
        self, client: EspireClient, config: dict, log_dir: Path, task: dict
    ) -> None:
        """
        Execute reflective localizing with iterative improvement cycles.

        Extends base localizing with reflection capabilities that analyze failed attempts
        and adapt strategies based on evaluation feedback. Maintains trial history
        for continuous performance optimization.

        Args:
            client (EspireClient): Client instance for server communication
            config (dict): Configuration parameters including reflection settings
            log_dir (Path): Directory for saving localizing attempts and reflections
            task (dict): Current task configuration dictionary
        """
        # Get observation for localizing
        observation = self.get_obs_for_localizing(client)

        # Parameters for reflection
        reflection = None
        prev_localizing_result = None

        # Begin loop
        localizing_result = None
        localizing_trials = config.get("localizing_trials", 3)
        for trial in range(localizing_trials):
            client.log(f"Localizing trial {trial + 1} / {localizing_trials}")

            # Localizing
            point = self.localizing(
                client,
                config,
                log_dir,
                task,
                observation,
                reflection=reflection,
                prev_localizing_result=prev_localizing_result,
            )
            localizing_result = self.execute_localizing(
                client, observation, point, trial
            )

            # Stop according to the result
            if localizing_result["goal_met"]:
                break

            # Reflect localizing
            if trial != localizing_trials - 1:
                reflection = self.reflect_localizing(
                    client, config, log_dir, task, localizing_result
                )
                prev_localizing_result = localizing_result

        if localizing_result is not None and not localizing_result["goal_met"]:
            client.localizing((0, 0), skip=True)

    @abstractmethod
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
        """
        Analyze moving execution results and generate adaptive improvement strategies.

        Processes moving outcome data including task completion status and
        precision metrics to develop optimized approaches for subsequent attempts.

        Args:
            client (EspireClient): Client instance for communication and logging
            config (dict): Configuration parameters for reflection analysis
            log_dir (Path): Directory for saving moving reflection data
            task (dict): Current task configuration for contextual adaptation
            obs (dict): Observation data from the moving attempt
            new_obs (dict): Observation data after the moving attempt
            moving_result (dict): Server response containing moving execution results

        Returns:
            str: Adaptive moving strategy insights for next attempt optimization

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        raise NotImplementedError("Reflect moving is not implemented!")

    @final
    def do_moving(
        self, client: EspireClient, config: dict, log_dir: Path, task: dict
    ) -> None:
        """
        Execute reflective moving with continuous strategy adaptation.

        Enhances base moving with reflection mechanisms that analyze previous
        attempt outcomes and adjust moving strategies dynamically. Maintains
        contextual memory across trials for progressive performance improvement.

        Args:
            client (EspireClient): Client instance for server communication
            config (dict): Configuration parameters including adaptive learning settings
            log_dir (Path): Directory for saving moving sequences and reflections
            task (dict): Current task configuration dictionary
        """
        # Get observation for the first moving
        observation = self.get_obs_for_first_moving(client)

        # Parameters for reflection
        reflection = None
        prev_observation = None
        prev_moving_result = None

        # Begin loop
        moving_trials = config.get("moving_trials", 5)
        for trial in range(moving_trials):
            client.log(f"Moving trial {trial + 1} / {moving_trials}")

            # Moving
            point = self.moving(
                client,
                config,
                log_dir,
                task,
                observation,
                reflection=reflection,
                prev_observation=prev_observation,
                prev_moving_result=prev_moving_result,
            )
            moving_result = self.execute_moving(client, observation, point, trial)

            # Stop according to the result
            if moving_result["goal_met"]:
                self.get_obs_for_moving(client)
                break

            # Prepare for the next moving
            if trial != moving_trials - 1:
                prev_observation = observation
                prev_moving_result = moving_result
                observation = self.get_obs_for_moving(client)

                # Reflect moving
                reflection = self.reflect_moving(
                    client,
                    config,
                    log_dir,
                    task,
                    prev_observation,
                    observation,
                    prev_moving_result,
                )


class TwoViewDriver(BaseDriver):
    """
    Moving with both world view and ego view

    Extends base driver with dual-view observation capabilities for enhanced
    spatial awareness and moving precision. Combines global context from
    world view with detailed egocentric perspective for improved task execution.

    Maintains synchronized observation streams from both camera perspectives
    to support complex spatial reasoning and multi-view coordination.
    """

    def get_obs_for_first_moving(self, client: EspireClient) -> dict:
        """
        Retrieve synchronized dual-view observations for initial moving phase.

        Captures both ego-centric and world-centric perspectives with goal masks
        enabled, providing comprehensive spatial context for moving planning.
        Ego view is designated as primary for visualization consistency.

        Args:
            client (EspireClient): Client instance for server communication

        Returns:
            dict: Combined observation data containing:
                - world_view (dict): Global perspective observation with goal mask
                - ego_view (dict): First-person perspective observation with goal mask
                - rgb_path (str): Primary image path for visualization (ego view)

        Raises:
            ConnectionError: If either observation request to server fails
        """
        try:
            ego_view = client.get_observation("ego", highlight_goal=True)["content"]
        except Exception as e:
            client.log(
                f"Failed to get observation from server for:\n{e}", level="ERROR"
            )
            raise ConnectionError

        try:
            world_view = client.get_observation("world", highlight_goal=True)["content"]
        except Exception as e:
            client.log(
                f"Failed to get observation from server for:\n{e}", level="ERROR"
            )
            raise ConnectionError

        return {
            "world_view": world_view,
            "ego_view": ego_view,
            "rgb_path": world_view["rgb_path"],  # used for drawing point
            "bbox_color": world_view["bbox_color"],
        }

    def get_obs_for_moving(self, client: EspireClient) -> dict:
        """
        Retrieve updated dual-view observations for subsequent moving iterations.

        Maintains continuous dual-perspective monitoring throughout moving sequence,
        enabling dynamic strategy adjustment based on real-time spatial changes.
        Preserves ego view as primary visualization source across all moving phases.

        Args:
            client (EspireClient): Client instance for server communication

        Returns:
            dict: Updated combined observation data with:
                - world_view (dict): Current global context with goal mask
                - ego_view (dict): Updated first-person perspective with goal mask
                - rgb_path (str): Active ego-view image path for visualization

        Raises:
            ConnectionError: If either observation request to server fails
        """
        try:
            world_view = client.get_observation("world", highlight_goal=True)["content"]
        except Exception as e:
            client.log(
                f"Failed to get observation from server for:\n{e}", level="ERROR"
            )
            raise ConnectionError

        try:
            ego_view = client.get_observation("ego", highlight_goal=True)["content"]
        except Exception as e:
            client.log(
                f"Failed to get observation from server for:\n{e}", level="ERROR"
            )
            raise ConnectionError

        return {
            "world_view": world_view,
            "ego_view": ego_view,
            "rgb_path": ego_view["rgb_path"],  # used for drawing point
            "bbox_color": ego_view["bbox_color"],
        }


# Type variable for driver implementations
EspireDriver = TypeVar("EspireDriver", bound=BaseDriver)
