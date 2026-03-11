import json
import uuid
from abc import ABC
from pathlib import Path
from typing import Any, Type, cast

import cv2
import msgpack
import zmq
from loguru import logger

from espire_eval_common.image import decode_color_png_rgb
from espire_eval_common.utils import remove_keys


class BaseClient(ABC):
    """
    Abstract base client for ZeroMQ-based communication with server.

    Provides core functionality for message serialization, logging, and connection
    management using context manager pattern for resource cleanup.

    Attributes:
        config (dict): Configuration parameters for client and server
        req_uid (str): Unique identifier for the current request
        log_dir (Path): Directory path for storing log files
        context (zmq.Context | None): ZeroMQ context instance
        sender (zmq.Socket | None): ZeroMQ socket for sending messages
        receiver (zmq.Socket | None): ZeroMQ socket for receiving messages
        MAX_VERIFY_ATTEMPTS (int): Maximum attempts for verifying (default: 10)
    """

    config: dict
    req_uid: str
    log_dir: Path
    context: zmq.Context | None
    sender: zmq.Socket | None
    receiver: zmq.Socket | None

    MAX_VERIFY_ATTEMPTS: int = 10

    def __init__(self, config: dict, log_dir: Path) -> None:
        """
        Initialize base client with configuration and logging directory.

        Args:
            config (dict): Configuration dictionary containing server/client settings
            log_dir (Path): Directory path for log file storage
        """
        self.config = config
        self.req_uid = ""
        self.log_dir = log_dir
        self.context = None
        self.sender = None
        self.receiver = None

    def __enter__(self) -> "BaseClient":
        """
        Set up ZeroMQ connections when entering context manager.

        Establishes PUSH socket for sending and SUB socket for receiving messages.

        Returns:
            BaseClient: Initialized client instance
        """
        logger.info("==== Client initializing ====")
        self.context = zmq.Context()

        # Configure sender socket (PUSH to server)
        server_url = (
            f"tcp://{self.config['server']['ip']}:{self.config['server']['port']}"
        )
        self.sender = self.context.socket(zmq.PUSH)
        assert self.sender is not None, "sender is None"
        self.sender.connect(server_url)
        logger.info(f"Connected to server at {server_url}")

        # Configure receiver socket (SUB from client port)
        client_url = (
            f"tcp://{self.config['client']['ip']}:{self.config['client']['port']}"
        )
        self.receiver = self.context.socket(zmq.SUB)
        assert self.receiver is not None, "receiver is None"
        self.receiver.connect(client_url)
        self.receiver.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        logger.info(f"Listening at {client_url}")

        logger.info("==== Client started ====")
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """
        Clean up resources when exiting context manager.

        Args:
            exc_type (Type[BaseException] | None): Exception type if raised in context
            exc_val (BaseException | None): Exception instance if raised
            exc_tb (Any): Traceback object if exception occurred

        Returns:
            bool: False to propagate exceptions, True to suppress
        """
        if exc_type is None:
            logger.info("==== Client stopped normally ====")
        else:
            logger.error(f"Client stopped due to {exc_type.__name__}: {exc_val}")

        self._cleanup()
        return False  # Propagate exceptions

    def _cleanup(self) -> None:
        """Safely close all ZeroMQ sockets and terminate context."""
        try:
            if self.sender:
                self.sender.close(linger=0)  # Non-blocking close
            if self.receiver:
                self.receiver.close(linger=0)
            if self.context:
                self.context.term()  # Terminate ZeroMQ context
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            self.sender = self.receiver = self.context = None
            logger.complete()
            logger.info("Resources cleaned up")

    def __del__(self) -> None:
        """Fallback cleanup for garbage collection."""
        if any([self.sender, self.receiver, self.context]):
            logger.warning("Client __del__ called: performing cleanup")
            self._cleanup()

    def log(
        self,
        content: str | None,
        data: dict | None = None,
        level: str = "INFO",
    ) -> None:
        """
        Log messages with optional structured data.

        Supports multiple log levels and handles both text content and JSON data.

        Args:
            content (str | None): Text message to log
            data (dict | None): Structured data to log as JSON
            level (str): Log level (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)

        Raises:
            ValueError: If unsupported log level is specified
        """
        # Prepare log message
        message_parts = []
        if content:
            message_parts.append(content)
        if data:
            message_parts.append(json.dumps(data, indent=2, ensure_ascii=False))

        message = "\n".join(message_parts) if message_parts else ""

        # Dispatch to appropriate log level
        log_levels = {
            "TRACE": logger.trace,
            "DEBUG": logger.debug,
            "INFO": logger.info,
            "SUCCESS": logger.success,
            "WARNING": logger.warning,
            "ERROR": logger.error,
            "CRITICAL": logger.critical,
        }

        log_func = log_levels.get(level.upper())
        if log_func is None:
            raise ValueError(f"Unsupported log level: {level}")

        log_func(message)

    def request(self, data: dict) -> dict:
        """
        Send request and wait for corresponding response.

        Handles message serialization, UUID generation, and response correlation.

        Args:
            body (dict): Request data to send

        Returns:
            dict: Response data from server
        """
        # Generate unique request ID
        self.req_uid = str(uuid.uuid4())
        request_data = {"uid": self.req_uid, "type": data.pop("type"), "content": data}

        # Log outgoing request
        self.log("Send", request_data)

        # Send serialized request
        assert self.sender is not None, "sender is None"
        self.sender.send(self.serialize(request_data))

        # Receive and correlate responses
        response_data = self._receive_correlated_response()
        response = response_data["response"]

        # Validate response
        self._validate_response(response)

        # Log incoming response
        self.log("Recv", response_data)

        return response

    def _receive_correlated_response(self) -> dict:
        """
        Receive messages until finding the one with matching request UID.

        Returns:
            dict: Deserialized response data with matching UID

        Raises:
            ValueError: If response uid has failed verification for MAX_VERIFY_ATTEMPTS consecutive times
            ValueError: If there is no `response` key in response
        """
        assert self.receiver is not None, "receiver is None"
        for _ in range(self.MAX_VERIFY_ATTEMPTS):
            response = self.receiver.recv()
            response = self.deserialize(response)

            if response.get("uid", None) == self.req_uid:
                if response.get("response", None) is not None:
                    return response
                else:
                    raise ValueError(f"There is no `response` key: {self.req_uid}")
            else:
                logger.warning(
                    f"The uid in the response ({response.get('uid', None)}) does not match the uid in the request ({self.req_uid})"
                )
        raise ValueError(
            f"The request uid in the response has failed verification for {self.MAX_VERIFY_ATTEMPTS} consecutive times."
        )

    def _validate_response(self, response: dict) -> None:
        """
        Validate response structure and success status.

        Args:
            response (dict): Response data to validate

        Raises:
            ValueError: If response indicates failure or has invalid structure
        """
        if response.get("code", 400) != 200:
            error_info = response.get("info", "Unknown error (no `info` in response)")
            if isinstance(error_info, str) and error_info.strip() == "":
                error_info = "Unknown error (has `info` but is empty)"
            raise ValueError(f"Request {self.req_uid} failed: {error_info}")

    @staticmethod
    def serialize(data: dict) -> bytes:
        """
        Serialize dictionary to bytes using MessagePack.

        Args:
            data (dict): Data to serialize

        Returns:
            bytes: Serialized binary data
        """
        return cast(bytes, msgpack.packb(data, use_bin_type=True))

    @staticmethod
    def deserialize(data: bytes) -> dict:
        """
        Deserialize bytes to dictionary using MessagePack.

        Args:
            data (bytes): Binary data to deserialize

        Returns:
            dict: Deserialized dictionary
        """
        return msgpack.unpackb(data, raw=False)


class EspireClient(BaseClient):
    """
    Specialized client for Espire

    Extends BaseClient with session management, observation data logging,
    and task execution methods for robotic manipulation tasks.

    Attributes:
        _session_uid (str): Unique session identifier
        last_obs_uid (str): UID of the last observation request
        OBS_KEYS (tuple): Keys identifying observation data types
    """

    _session_uid: str
    last_obs_uid: str

    OBS_KEYS: set[str] = set(["rgb", "depth", "intrinsic", "extrinsic"])

    def __init__(self, config: dict, log_dir: Path) -> None:
        """
        Initialize Espire client with session management.

        Args:
            config (dict): Configuration parameters
            log_dir (Path): Log directory path
        """
        super().__init__(config, log_dir)
        self._session_uid = ""
        self.last_obs_uid = ""

    def __enter__(self) -> "EspireClient":
        """
        Set up observation directories and session ID.

        Returns:
            EspireClient: Initialized client instance
        """
        super().__enter__()

        # Create observation data directories
        for data_type in self.OBS_KEYS:
            Path(self.log_dir / data_type).mkdir(exist_ok=True)

        self._session_uid = str(uuid.uuid4())
        return self

    def log(
        self,
        content: str | None,
        data: dict | None = None,
        level: str = "INFO",
    ) -> None:
        """
        Enhanced logging with observation data extraction and storage.

        Handles RGB images, depth maps, and calibration data separately from log messages.

        Args:
            content (str | None): Log message content
            data (dict | None): Data payload potentially containing observation data
            level (str): Logging level (INFO, DEBUG, WARNING, etc.)
        """
        if data:
            # Extract and save observation data
            self._save_observation_data(data)

            # Log remaining data (excluding observation fields)
            log_data = remove_keys(data, self.OBS_KEYS)
            super().log(content, log_data, level)
        else:
            super().log(content, None, level)

    def _save_observation_data(self, data: dict) -> None:
        """
        Save observation data (images, depth, calibration) to appropriate files.

        Args:
            data (dict): Data dictionary potentially containing observation fields
        """
        req_uid = data.get("uid", self.req_uid)
        content = data.get("response", {}).get("content", {})

        if "rgb" in content:
            content["rgb_path"] = self._save_rgb_image(content["rgb"], req_uid)
        if "depth" in content:
            content["depth_path"] = self._save_binary_data(
                content["depth"], "depth", req_uid
            )
        if "intrinsic" in content:
            content["intrinsic_path"] = self._save_binary_data(
                content["intrinsic"], "intrinsic", req_uid
            )
        if "extrinsic" in content:
            content["extrinsic_path"] = self._save_binary_data(
                content["extrinsic"], "extrinsic", req_uid
            )

    def _save_rgb_image(self, rgb_data: bytes, request_uid: str) -> str:
        """
        Save RGB image data as PNG file.

        Args:
            rgb_data (bytes): Compressed RGB image data
            request_uid (str): Unique identifier for the request

        Returns:
            str: Saved path of RGB image

        Raises:
            OSError: If file writing fails
        """
        image_path = self.log_dir / "rgb" / f"{request_uid}.png"
        try:
            # Decode PNG and convert RGB to BGR for OpenCV
            rgb_image = decode_color_png_rgb(rgb_data)
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            cv2.imwrite(
                str(image_path.resolve()), bgr_image, [cv2.IMWRITE_PNG_COMPRESSION, 5]
            )
        except Exception as e:
            logger.error(f"Failed to save RGB image for request {request_uid}: {e}")
        return str(image_path.resolve())

    def _save_binary_data(self, data: bytes, data_type: str, request_uid: str) -> str:
        """
        Save binary observation data to file.

        Args:
            data (bytes): Binary data to save
            data_type (str): Type of data (depth, intrinsic, extrinsic)
            request_uid (str): Unique identifier for the request

        Returns:
            str: Saved path of binary data

        Raises:
            OSError: If file writing fails
        """
        file_path = self.log_dir / data_type / f"{request_uid}.bin"
        try:
            file_path.write_bytes(data)
        except Exception as e:
            logger.error(
                f"Failed to save {data_type} data for request {request_uid}: {e}"
            )
        return str(file_path.resolve())

    @property
    def session_uid(self) -> str:
        """Get current session unique identifier."""
        return self._session_uid

    # Task management methods
    def get_task_list(self) -> dict:
        """
        Retrieve available tasks from the server.

        Returns:
            dict: Server response containing available tasks
        """
        request = {"type": "GET_TASK_LIST"}
        return self.request(request)

    def set_task(
        self, task_id: str, rotation: tuple[str | None, str | None, str | None]
    ) -> dict:
        """
        Set current task with rotation requirement.

        Args:
            task_id (str): Unique task identifier
            rotation (tuple[str | None, str | None, str | None]): Rotation requirement specification

        Returns:
            dict: Server response
        """
        request = {
            "type": "SET_TASK",
            "task_id": task_id,
            "rotation": rotation,
        }
        return self.request(request)

    def get_observation(self, view_name: str, highlight_goal: bool) -> dict:
        """
        Request observation data from specified camera view.

        Args:
            view_name (str): Name of the camera view to capture (world | ego)
            highlight_goal (bool): Whether to highlight goal mask in observation

        Returns:
            dict: Observation data containing images and calibration information
        """
        request = {
            "type": "GET_OBSERVATION",
            "view": view_name,
            "highlight_goal": highlight_goal,
        }
        response = self.request(request)
        self.last_obs_uid = self.req_uid
        return response

    def localizing(self, point: tuple[int, int], skip: bool = False) -> dict:
        """
        Perform localizing operation at specified point coordinates.

        Args:
            point (tuple[int, int]): (row, col) point coordinates for localization
            skip (bool): Whether to skip actual execution

        Returns:
            dict: Localizing result with evaluation metrics
        """
        if skip:
            request = {"type": "SKIP_LOCALIZATION", "goal": None}
        else:
            request = {
                "type": "LOCALIZE",
                "reference_uid": self.last_obs_uid,
                "point": {"row": point[0], "col": point[1]},
            }
        return self.request(request)

    def move_to(
        self,
        point: tuple[int, int],
        skip: bool = False,
        rotation_delta: tuple[float | None, float | None, float | None] | None = None,
    ) -> dict:
        """
        Execute movement command of specified type.

        Args:
            point (tuple[int, int]): (row, col) point coordinates for movement
            skip (bool): Whether to skip actual execution
            rotation_delta (tuple | None): Rotation delta for movement

        Returns:
            dict: Movement execution result
        """
        if skip:  # just end task
            return self.end_task()

        request = {
            "type": "MOVE",
            "reference_uid": self.last_obs_uid,
            "point": {"row": point[0], "col": point[1]},
        }
        if rotation_delta is not None:
            request["rotation_delta"] = rotation_delta

        return self.request(request)

    def end_task(self) -> dict:
        """
        Signal task completion to server.

        Returns:
            dict: Server response confirming task termination
        """
        request = {"type": "END_TASK"}
        return self.request(request)

    def request(self, data: dict) -> dict:
        """
        Send request with session context to server.

        Args:
            data (dict): Request data to send

        Returns:
            dict: Response data from server
        """
        data["sid"] = self.session_uid
        return super().request(data)
