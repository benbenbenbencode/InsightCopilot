import streamlit as st
from core.pipeline import run_pipeline
from app.chart import render_chart

st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.title("📊 AI Data Analyst (Local)")

question = st.text_input("Ask your data")

if st.button("Run"):
    result = run_pipeline(question)

    st.subheader("🧠 Generated SQL")
    st.code(result.get("sql", ""), language="sql")

    # ❌ ERROR
    if "error" in result:
        st.error("Query failed")
        st.text(result.get("error", "Unknown error"))

    # ✅ SUCCESS
    elif "data" in result:
        df = result["data"]

        st.subheader("📊 Data")
        st.dataframe(df)

        # 🔥 ADD CHART HERE
        render_chart(df)

        # 💡 Insights
        if "explanation" in result:
            st.subheader("💡 Insights")
            st.write(result["explanation"])

    else:
        st.warning("No result returned")