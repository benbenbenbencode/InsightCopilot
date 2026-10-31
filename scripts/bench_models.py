"""Benchmark installed Ollama models on a small NL->SQL task.

Measures wall time, tokens/sec, and prints the SQL so quality can be judged.
This is the evidence behind the default model choice in README.md.

    python -m scripts.bench_models
"""

import json
import time
import urllib.request

OLLAMA = "http://127.0.0.1:11434"

SCHEMA = """
fct_transactions(order_id text, user_id text, product_id text, order_date date, city text, quantity bigint, price bigint, revenue bigint)
dim_users(user_id text, city text, signup_date date)
dim_products(product_id text, product_name text, category text)
fct_promotions(promo_date timestamp, product_id text, promo_cost bigint)

Join keys:
  fct_transactions.user_id = dim_users.user_id
  fct_transactions.product_id = dim_products.product_id
  fct_promotions.product_id = dim_products.product_id
"""

QUESTIONS = [
    "Show total revenue by product category, highest first.",
    "Which city generated the most revenue per user?",
]

SYSTEM = (
    "You are a PostgreSQL analyst. Reply with one SELECT query only, no prose, "
    "no markdown fences. Use only the tables and columns given."
)

MODELS = [
    "qwen2.5-coder:7b",
    "hf.co/hotdogs/Qwen3.8-27B-abliterated-cyber-preview-MTP-GGUF:Q4_K_M",
    "qwen3.6:latest",
]


def ask(model, question, timeout=600):
    prompt = f"{SYSTEM}\n\nSchema:\n{SCHEMA}\n\nQuestion: {question}\nSQL:"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": "2m",
        "options": {"temperature": 0, "num_predict": 260, "num_ctx": 4096},
    }
    request = urllib.request.Request(
        f"{OLLAMA}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read())
    wall = time.time() - started

    tokens = data.get("eval_count") or 0
    duration = data.get("eval_duration") or 0
    rate = tokens / (duration / 1e9) if duration else 0
    return wall, tokens, rate, (data.get("response") or "").strip()


def unload(model):
    try:
        payload = {"model": model, "keep_alive": 0, "prompt": ""}
        request = urllib.request.Request(
            f"{OLLAMA}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=60).read()
    except Exception:
        pass


def main() -> None:
    results = []
    for model in MODELS:
        print(f"\n=== {model} ===", flush=True)
        for question in QUESTIONS:
            try:
                wall, tokens, rate, sql = ask(model, question)
                print(f"  Q: {question}", flush=True)
                print(f"  {wall:.1f}s  {tokens} tok  {rate:.1f} tok/s", flush=True)
                print("  " + sql.replace("\n", "\n  ")[:600], flush=True)
                results.append(
                    {"model": model, "question": question, "wall": round(wall, 1), "rate": round(rate, 1)}
                )
            except Exception as exc:
                print(f"  FAILED after {exc}", flush=True)
                results.append({"model": model, "question": question, "error": str(exc)[:120]})
        unload(model)

    print("\n=== SUMMARY ===")
    for row in results:
        if "error" in row:
            print(f"{row['model'][:40]:<42} ERROR {row['error']}")
        else:
            print(f"{row['model'][:40]:<42} {row['wall']:>6.1f}s {row['rate']:>6.1f} tok/s")


if __name__ == "__main__":
    main()
