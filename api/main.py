from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from core.anomaly import detect_anomalies
from core.pipeline import run_pipeline
from core.insight import generate_insight
from core.recommender import generate_recommendations
from core.root_cause import analyze_root_cause

app = FastAPI(title="AI Data Analyst API")


# 📥 request schema
class QueryRequest(BaseModel):
    question: str


# 🏠 health check
@app.get("/")
def root():
    return {"message": "AI Data Analyst API running"}


# 🚀 main endpoint
@app.post("/analyze")
def analyze(req: QueryRequest):
    print("🔥 FINAL VERSION RUNNING 🔥")

    result = run_pipeline(req.question)

    # ❌ error case
    if "error" in result:
        return {
            "status": "error",
            "error": result["error"]
        }

    # 🔥 convert to DataFrame
    data = result.get("data")
    df = pd.DataFrame(data) if data is not None else None

    # 🔥 normalize columns
    if df is not None:
        df = df.rename(columns={
            "product_name": "product",
            "revenue": "sales",
            "order_date": "date"
        })

    # 🔥 insight (LLM, context-aware)
    try:
        insight = generate_insight(df, req.question) if df is not None else None
    except Exception as e:
        print("INSIGHT ERROR:", str(e))
        insight = None

    # 📦 base response
    response = {
        "sql": result.get("sql"),
        "explanation": insight,
        "data": df.to_dict(orient="records") if df is not None else []
    }

    # 🚨 anomaly detection
    try:
        anomalies = detect_anomalies(df)

        if not anomalies.empty:
            response["anomalies"] = anomalies.to_dict(orient="records")
        else:
            response["anomalies"] = []

    except Exception as e:
        print("ANOMALY ERROR:", str(e))
        response["anomalies"] = []
        response["anomalies_error"] = str(e)

    # 🔍 root cause analysis
    try:
        if "anomalies" in response and response["anomalies"]:
            anomaly_df = pd.DataFrame(response["anomalies"])
            root_causes = analyze_root_cause(df, anomaly_df)
            response["root_cause"] = root_causes
        else:
            response["root_cause"] = []

    except Exception as e:
        print("ROOT CAUSE ERROR:", str(e))
        response["root_cause"] = []
        response["root_cause_error"] = str(e)    

    # 🚀 recommendation (from core)
    try:
        recs = generate_recommendations(df)
        response["recommendations"] = recs.to_dict(orient="records")
    except Exception as e:
        print("RECOMMENDER ERROR:", str(e))
        response["recommendations"] = []
        response["recommendations_error"] = str(e)

    return response