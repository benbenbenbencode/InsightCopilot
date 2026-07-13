import re
import pandas as pd

from llm.deepseek import call_llm


def _numbers_in_df(df: pd.DataFrame) -> set:
    """Collects numbers that are legitimately derivable from the dataframe:
    every raw cell value, plus each numeric column's sum/mean/count, since
    "the total is X" is a completely normal and correct thing for an insight
    to say even though X itself isn't a raw cell.
    """
    values = set()
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        for v in series.tolist():
            values.add(round(float(v)))
        if len(series):
            values.add(round(float(series.sum())))
            values.add(round(float(series.mean())))
            values.add(int(series.count()))
    return values


def _numbers_in_text(text: str) -> set:
    found = re.findall(r"-?\d[\d,]*\.?\d*", text)
    values = set()
    for f in found:
        cleaned = f.replace(",", "")
        try:
            values.add(round(float(cleaned)))
        except ValueError:
            continue
    return values


def generate_insight(df: pd.DataFrame, question: str) -> str:
    if df is None or df.empty:
        return "No data to analyze."

    df = df.copy()

    cols = [col for col in ["product", "sales", "date"] if col in df.columns]
    df_for_prompt = df[cols] if cols else df

    sample_n = min(len(df_for_prompt), 20)
    sample = df_for_prompt.head(sample_n).to_string(index=False)

    prompt = f"""
You are a senior data analyst.

User question:
{question}

Dataset (showing {sample_n} of {len(df_for_prompt)} rows):
{sample}

Instructions:
- Answer using ONLY the numbers shown above. Do not invent or estimate numbers
  that are not in the data.
- Give 3-5 insights as bullet points.
- If the sample above is small relative to the full result set, or the data
  looks insufficient to support a strong claim, say so explicitly instead of
  overstating confidence.
- End with one sentence recommending a concrete next action, only if the
  data actually supports one.

Format:
- Bullet points
"""

    try:
        response = call_llm(prompt).strip()
    except Exception as e:
        return f"Insight generation failed: {str(e)}"

    # --- Grounding check -------------------------------------------------
    # Cheap guardrail: flag numbers the model mentioned that don't appear
    # anywhere in the sampled data. This won't catch every hallucination
    # (the model can still misinterpret real numbers), but it catches the
    # common case of invented figures.
    claimed = _numbers_in_text(response)
    real = _numbers_in_df(df_for_prompt)
    suspicious = {n for n in claimed if n not in real and abs(n) > 1}

    if suspicious:
        response += (
            "\n\n_Note: this insight mentions figure(s) "
            f"{sorted(suspicious)} that don't match the sampled data exactly "
            "-- verify against the full result set before relying on them._"
        )

    if sample_n < len(df_for_prompt):
        response += (
            f"\n\n_Based on a sample of {sample_n} of {len(df_for_prompt)} "
            "rows, not the full result set._"
        )

    return response
