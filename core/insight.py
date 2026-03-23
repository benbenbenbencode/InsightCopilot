from llm.deepseek import call_llm
import pandas as pd


def generate_insight(df: pd.DataFrame, question: str) -> str:
    if df is None or df.empty:
        return "No data to analyze."

    df = df.copy()

    # 🔥 keep only important columns (reduce noise)
    cols = [col for col in ["product", "sales", "date"] if col in df.columns]
    df = df[cols]

    # 🔥 reduce size (important for LLM)
    sample = df.head(20).to_string(index=False)

    prompt = f"""
You are a senior data analyst.

You MUST analyze the data provided. Do NOT say you cannot analyze.

User question:
{question}

Dataset:
{sample}

Instructions:
- Answer based ONLY on the dataset
- Give 3–5 insights
- Be concise
- Use bullet points
- Do NOT mention limitations


Format:
- Bullet points
"""

    try:
        response = call_llm(prompt)
        return response.strip()

    except Exception as e:
        return f"Insight generation failed: {str(e)}"