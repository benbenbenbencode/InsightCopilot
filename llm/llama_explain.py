import subprocess


def explain(df, question: str):
    sample = df.head(20).to_csv(index=False)

    prompt = f"""
You are a data analyst.

Question:
{question}

Data:
{sample}

Explain insights briefly.
"""

    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="ignore"
    )

    return result.stdout.strip()