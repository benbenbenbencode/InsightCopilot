import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content.strip()