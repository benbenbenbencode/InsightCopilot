import subprocess


def explain(sql: str, df_preview: str) -> str:
    prompt = f"""
Explain this SQL result in simple business terms.

SQL:
{sql}

Data Preview:
{df_preview}

Explain:
- what it shows
- key insight
"""

    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        text=True,
        capture_output=True
    )

    return result.stdout.strip()