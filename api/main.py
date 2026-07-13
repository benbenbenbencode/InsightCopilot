from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from core.anomaly import detect_anomalies
from core.pipeline import run_pipeline
from core.insight import generate_insight
from core.recommender import generate_recommendations
from core.root_cause import analyze_root_cause

app = FastAPI(title="AI Data Analyst API")


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "AI Data Analyst API running"}


@app.post("/analyze")
def analyze(req: QueryRequest):
    result = run_pipeline(req.question)

    if "error" in result:
        return {"status": "error", "error": result["error"]}

    data = result.get("data")
    df = pd.DataFrame(data) if data is not None else None

    # These are the only three columns the current anomaly/recommender/
    # root-cause modules understand. If a query doesn't return them (e.g. a
    # pure by-city aggregation with no date column), those steps are
    # skipped explicitly below rather than failing silently.
    if df is not None:
        df = df.rename(columns={
            "product_name": "product",
            "revenue": "sales",
            "order_date": "date",
        })

    try:
        insight = generate_insight(df, req.question) if df is not None else None
    except Exception as e:
        print("INSIGHT ERROR:", str(e))
        insight = None

    response = {
        "sql": result.get("sql"),
        "explanation": insight,
        "data": df.to_dict(orient="records") if df is not None else [],
    }

    has_required_cols = df is not None and {"sales", "date"}.issubset(df.columns)

    if not has_required_cols:
        response["anomalies"] = []
        response["anomalies_skipped_reason"] = (
            "query result has no date/sales columns to analyze" if df is not None else None
        )
    else:
        try:
            anomalies = detect_anomalies(df, value_col="sales", date_col="date")
            response["anomalies"] = anomalies.to_dict(orient="records") if not anomalies.empty else []
        except Exception as e:
            print("ANOMALY ERROR:", str(e))
            response["anomalies"] = []
            response["anomalies_error"] = str(e)

    if not has_required_cols or "product" not in (df.columns if df is not None else []):
        response["root_cause"] = []
    elif response.get("anomalies"):
        try:
            anomaly_df = pd.DataFrame(response["anomalies"])
            root_causes = analyze_root_cause(
                df, anomaly_df, value_col="sales", date_col="date", dimension_col="product"
            )
            response["root_cause"] = root_causes
        except Exception as e:
            print("ROOT CAUSE ERROR:", str(e))
            response["root_cause"] = []
            response["root_cause_error"] = str(e)
    else:
        response["root_cause"] = []

    if df is not None and "product" in df.columns and "sales" in df.columns:
        try:
            recs = generate_recommendations(df, dimension_col="product", value_col="sales")
            response["recommendations"] = recs.to_dict(orient="records")
        except Exception as e:
            print("RECOMMENDER ERROR:", str(e))
            response["recommendations"] = []
            response["recommendations_error"] = str(e)
    else:
        response["recommendations"] = []

    return response
