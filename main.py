import argparse
from pathlib import Path

from espire_eval_common.analyze import analyze
from espire_eval_common.start import start

from gemini_wrapper.driver import EspireDriver
from utils.file import proj_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EspireDriver evaluation")

    parser.add_argument(
        "--config",
        type=Path,
        default=proj_dir() / "config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )

    parser.add_argument(
        "--test-grounding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable grounding test (default: enabled)",
    )

    parser.add_argument(
        "--test-moving",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable moving test (default: enabled)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="Number of execution cycles (default: 300)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=-1,
        help="Timeout in seconds, -1 means no timeout (default: -1)",
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analysis after evaluation (default: disabled)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    start(
        driver_cls=EspireDriver,
        config_path=args.config,
        test_grounding=args.test_grounding,
        test_moving=args.test_moving,
        iterations=args.iterations,
        timeout=args.timeout,
    )

    if args.analyze:
        analyze(proj_dir() / "log")
