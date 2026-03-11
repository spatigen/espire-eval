import multiprocessing
import time
from datetime import datetime
from pathlib import Path
from typing import Type

import yaml
from loguru import logger

from espire_eval_common.driver import EspireDriver


def set_logger(config: dict) -> Path:
    """
    Configure and initialize the application logger.

    Creates a timestamped log directory, sets up log file rotation with compression,
    and configures log format and levels. Uses loguru for advanced logging features.

    Args:
        config (dict): Configuration dictionary containing logging settings

    Returns:
        Path: Path object pointing to the created log directory
    """
    # Create timestamped log directory
    log_dir = Path(config["client"]["log_dir"]) / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing handlers and configure new file handler
    logger.remove()

    # Configure log file with rotation and compression
    logger.add(
        log_dir / "run.log",
        level="INFO",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        backtrace=True,
        diagnose=False,  # Disable detailed traceback to avoid sensitive data exposure
        enqueue=True,  # Thread-safe logging
        rotation="100 MB",
        compression="tar.gz",
        mode="w",  # Overwrite on each run for clean logs
    )

    return log_dir


def start(
    driver_cls: Type[EspireDriver],
    config_path: str | Path,
    test_grounding: bool = True,
    test_moving: bool = True,
    iterations: int = 300,
    timeout: int = -1,
) -> None:
    """
    Main execution loop for running driver instances with timeout protection.

    Loads configuration, initializes driver, and executes multiple iterations
    with optional timeout monitoring using separate processes.

    Args:
        driver_cls (EspireDriver): Driver class implementing the EspireDriver interface
        config_path (str | Path): Path to YAML configuration file
        test_grounding (bool): If True, test grounding (True by default)
        test_moving (bool): If True, test moving (True by default)
        iterations (int): Number of execution cycles (default: 300)
        timeout (int): Maximum execution time in seconds, -1 for no timeout (default: -1)

            If enabled (timeout > 0), the driver will be executed in a separate process
            using the ``multiprocessing`` module, allowing the main process to terminate the
            worker process if it exceeds the timeout limit.

            Note: The subprocess is started using the default method.
            If the driver process uses CUDA, the multiprocessing start method must be set to ``spawn``.
            However, this does not interoperate very well with Loguru's default logging mechanism (see
            https://loguru.readthedocs.io/en/stable/resources/recipes.html#compatibility-with-multiprocessing-using-enqueue-argument),
            the logging output may behave unexpectedly.
    """
    # Load configuration with explicit encoding specification
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Initialize driver instance
    driver = driver_cls(config["client"]["log_dir"])

    # Execute specified number of iterations
    for i in range(iterations):
        log_dir = set_logger(config)
        logger.info(f"Starting iteration: {i}")

        # Execute with or without timeout monitoring
        if timeout == -1:
            driver.run(config, log_dir, test_grounding, test_moving)
        else:
            # Use separate process for timeout control
            process = multiprocessing.Process(
                target=driver.run, args=(config, log_dir, test_grounding, test_moving)
            )
            process.start()
            process.join(timeout=timeout)

            # Handle timeout termination
            if process.is_alive():
                logger.error(
                    f"Terminating iteration {i} due to timeout: {timeout} seconds"
                )
                process.terminate()
                process.join(timeout=5)  # Brief grace period

                # Force kill if still alive after termination
                if process.is_alive():
                    process.kill()
                    logger.warning("Process required forced termination")

        # Sleep 0.05 second
        time.sleep(0.05)
