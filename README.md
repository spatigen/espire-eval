# Espire Evaluation Module: RoboBrain Wrapper

## 🚀 Quick Start
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Setup:
    ```bash
    # Clone repository
    git clone -b robobrain https://github.com/spatigen/espire-eval.git

    cd espire-eval

    # Init and Sync
    uv init
    uv sync

    # Run
    uv run main.py
    ```

---

## 🖥️ CLI Help

```bash
usage: main.py [-h] [--config CONFIG] [--test-grounding | --no-test-grounding] [--test-moving | --no-test-moving]
               [--iterations ITERATIONS] [--timeout TIMEOUT] [--analyze]

Run EspireDriver evaluation

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to YAML configuration file (default: config.yaml)
  --test-grounding, --no-test-grounding
                        Enable grounding test (default: enabled)
  --test-moving, --no-test-moving
                        Enable moving test (default: enabled)
  --iterations ITERATIONS
                        Number of execution cycles (default: 300)
  --timeout TIMEOUT     Timeout in seconds, -1 means no timeout (default: -1)
  --analyze             Run analysis after evaluation (default: disabled)
```

---

## ℹ️ Model Notes

Tested [RoboBrain2.0-7B](https://huggingface.co/BAAI/RoboBrain2.0-7B)
