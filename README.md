# Espire Evaluation Module: Qwen Wrapper

## 🚀 Quick Start
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Setup:
    ```bash
    # Clone repository
    git clone -b qwen https://github.com/spatigen/espire-eval.git

    cd espire-eval

    # Init and Sync
    uv init
    uv sync

    # Set your llm api key if not
    # export QWEN_API_KEY="YOUR_API_KEY"

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

Coordinate System:
- Qwen2.5-VL returns coordinates as absolute pixel values relative to the top-left corner of the resized image.
- Qwen3-VL's default coordinate system has been changed from the absolute coordinates used in Qwen2.5-VL to relative coordinates ranging from 0 to 1000. (You don't need to calculate the resized_w)

---

## 🧾 Prompt Tips

|Localization Method|Supported Output Format|Recommended Prompt|
|:---|:---|:---|
|Box|JSON or Plain Text|Detect all {objects} in the image and output their bbox coordinates in {JSON/plain text} format|
|Point|JSON or XML|Locate all {objects} in the image using points, and output their point coordinates in {JSON/XML} format|

[Examples](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb)
