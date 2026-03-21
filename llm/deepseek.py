import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-coder"  # or your model name


def call_llm(prompt: str):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()
    return data.get("response", "")