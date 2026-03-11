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

模型支持对物体进行定位：
- Qwen2.5-VL 模型返回的坐标均相对于缩放后的图像左上角的绝对值，单位为像素。可参考 Qwen2.5-VL 中的代码将坐标映射到原图中。
- Qwen2.5-VL 模型 480\*480 ~ 2560\*2560 分辨率范围内，物体定位效果较为鲁棒，在此范围之外可能会偶发 bbox 漂移现象。
- Qwen3-VL 模型返回的坐标将为相对坐标，坐标值会归一化到 0-999。

---

## 🧾 Prompt Tips

|定位方式|支持的输出方式|推荐Prompt|
|:---|:---|:---|
|Box 定位|JSON 或纯文本|检测图中所有{物体}并以{JSON/纯文本}格式输出其 bbox 的坐标|
|Point 定位|JSON 或 XML|以点的形式定位图中所有{物体}，以{JSON/XML}格式输出其 point 坐标|

[Examples](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb)
