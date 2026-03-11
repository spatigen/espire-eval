import json
import time

import requests


def call_internvl(
    model_name: str,
    api_key: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    max_attempts: int = 5,
    sleep_interval: int = 10,
) -> str:
    error_info = ""
    for attempt in range(max_attempts):
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
                # "X-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
            },
            data=json.dumps(
                {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            ),
        )
        if response.status_code == 200:
            res = json.loads(response.text)
            return res["choices"][0]["message"]["content"]
        else:
            error_info += f"[Attempt {attempt}] failed: {response.text}\n"
            time.sleep(sleep_interval)

    raise Exception(f"API request failed {max_attempts} times:\n{error_info}")
