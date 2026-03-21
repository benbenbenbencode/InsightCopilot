import streamlit as st
from core.pipeline import run_pipeline
from app.charts import render_chart


st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.title("📊 AI Data Analyst")

# =========================
# USER INPUT
# =========================
question = st.text_input("Ask your data question:")

if st.button("Run"):

    if not question:
        st.warning("Please enter a question")
        st.stop()

    with st.spinner("Analyzing..."):
        result = run_pipeline(question)

    # =========================
    # ERROR
    # =========================
    if "error" in result:
        st.error(result["error"])
        st.code(result["sql"], language="sql")
        st.stop()

    # =========================
    # SQL DEBUG
    # =========================
    with st.expander("🧠 Generated SQL"):
        st.code(result["sql"], language="sql")

    # =========================
    # DATA + CHART
    # =========================
    df = result["data"]

    if df is not None and not df.empty:
        st.subheader("📄 Data")
        st.dataframe(df)

        render_chart(df)

    else:
        st.warning("No data returned")

    # =========================
    # AI INSIGHT
    # =========================
    if "insight" in result:
        st.subheader("🧠 Insight")
        st.write(result["insight"])