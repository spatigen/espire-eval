# Espire Evaluation

## 🚀 Quick Start

0. Server Setup: Before running the evaluation, you need to [set up](https://github.com/spatigen/espire) and start the server:
    ```bash
    cd $SERVER_PROJECT_PATH && docker compose run --rm --service-ports espire bash scripts/serve.sh
    ```
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Setup:
    ```bash
    # Clone repository
    git clone -b <MODEL> https://github.com/spatigen/espire-eval.git

    cd espire-eval

    # Init and Sync
    uv init
    uv sync

    # Set your llm api key if not
    # export <MODEL>_API_KEY="YOUR_API_KEY"

    # Run
    uv run main.py
    ```
> Replace \<MODEL\> with one of the following options: `qwen`, `gemini`, `internvl`, `robobrain` (w/o api key)

---

## 🛠️ Custom Model Integration

To integrate your own model into the Espire Evaluation framework, follow these steps for a minimal setup:

### 1. Implement the Driver
In your implementation file, create a new class (recommended name: `EspireDriver`) that inherits from the `BaseDriver` class found in [src/espire_eval_common/driver.py](src/espire_eval_common/driver.py#L12).

You are required to implement the following two methods, which correspond to the two critical stages of the evaluation:
- `localizing`: Define how your model handles the localization phase.
- `moving`: Define the control logic for the execution/movement phase.

### 2. Launch the Evaluation
To run the evaluation with your custom model, use the `start` function provided in [src/espire_eval_common/start.py](src/espire_eval_common/start.py#L51).

**Example Workflow:**
- Create a `main.py` (you can refer to the `main.py` in existing model branches).
- Import your `EspireDriver`.
- Pass the driver and necessary parameters into the `start` function.

### Advanced Integration

The [src/espire_eval_common/driver.py](src/espire_eval_common/driver.py) file provides additional driver types for more complex scenarios:

* [ReflectionDriver](src/espire_eval_common/driver.py#L467): Use this if your model supports a reflective execution flow. You will need to implement additional `NotImplemented` methods to define the specific logic for reflection.
* [TwoViewDriver](src/espire_eval_common/driver.py#L666): Use this if your testing process requires simultaneous access to both **World View** and **Ego View**.

---

## 🤖 Supported Model

| Model | Branch | URL |
|------|------|------|
| Qwen | `qwen` | https://github.com/spatigen/espire-eval/tree/qwen |
| Gemini | `gemini` | https://github.com/spatigen/espire-eval/tree/gemini |
| InternVL | `internvl` | https://github.com/spatigen/espire-eval/tree/internvl |
| RoboBrain | `robobrain` | https://github.com/spatigen/espire-eval/tree/robobrain |

---

## 🧠 Pseudocode for Reflective Execution

### 🔎 Localization with Reflection
<img src="assets/reflective_localize.png" width="70%" />

### 🧭 Execution with Reflection
<img src="assets/reflective_execution.png" width="70%" />
