from llm.deepseek import call_llm

def generate_insight(df, question):
    if df is None or df.empty:
        return "No data to analyze."

    # 🔥 reduce size (VERY IMPORTANT)
    sample = df.head(20).to_string()

    prompt = f"""
You are a senior data analyst.

User question:
{question}

Here is the aggregated data:
{sample}

Give 3-5 short business insights.
Focus on trends, comparison, and key takeaways.
Be concise.
"""

    return call_llm(prompt)