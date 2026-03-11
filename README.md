# Espire Evaluation Module: InternVL Wrapper

## 🚀 Quick Start
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Setup:
    ```bash
    # Clone repository
    git clone -b internvl https://github.com/spatigen/espire-eval.git

    cd espire-eval

    # Init and Sync
    uv init
    uv sync

    # Set your llm api key if not
    # export INTERNVL_API_KEY="YOUR_API_KEY"

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

## 🧾 Prompt Tips

- [Grounding / Detection Data](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#grounding-detection-data)
- [evaluate_grounding.py](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/eval/refcoco/evaluate_grounding.py)
- [Issue 1140](https://github.com/OpenGVLab/InternVL/issues/1140)
