# 🧠 AI Data Analyst (Local)

An end-to-end AI-powered data analyst that transforms natural language into SQL, executes queries, and automatically generates insights, root causes, and actionable business recommendations.

---

## 🚀 Why This Project Matters

Most dashboards only show **what happened**.

This system goes further:
- ✅ Explains **why it happened**
- ✅ Detects **performance issues**
- ✅ Suggests **what to do next**

👉 Simulating a real **AI Business Analyst**

---

## ✨ Key Features

- 🔍 Natural language → SQL (LLM-powered)
- 🔗 Automatic table joins (semantic join engine)
- 📊 Smart visualization (Altair + Streamlit)
- 🧠 Insight generation (top performer, gap, efficiency)
- 🚨 Anomaly detection (statistical patterns)
- 🔍 Root cause analysis (ROI & revenue behavior)
- 🚀 Business recommendations (actionable insights)

---

## 🏗️ System Architecture
User Question
↓
LLM (generate SQL)
↓
Pipeline (validation + join engine)
↓
Postgres DB
↓
Pandas DataFrame
↓
Chart + Insights + Root Cause + Recommendation

---

## 📂 Project Structure
app/
chart.py # visualization + insights + recommendation
ui.py # UI components
chatbot.py # user interaction

core/
pipeline.py # main orchestration logic
join_engine.py # automatic join resolution
sql_validator.py

db/
postgres.py # database connection + caching

llm/
deepseek_sql.py # SQL generation via LLM
insight.py # insight & reasoning layer

retail_dbt/
models/ # dbt models (facts & dimensions)


## ⚙️ Tech Stack

- **Python** (Pandas, Streamlit)
- **PostgreSQL**
- **dbt** (data modeling)
- **Altair** (visualization)
- **Ollama / DeepSeek** (LLM)

---

## 🧪 Example Use Cases

- "Compare total revenue and promo cost by city"
- "Which product has the highest ROI?"
- "Show performance differences across regions"

---

## ⚡ Performance Optimization

- ✅ Query caching (`@st.cache_data`)
- ✅ Pipeline caching
- ✅ Aggregation-first SQL strategy
- ✅ Reduced data transfer (LIMIT)

---

## 🧠 Example Output

Instead of raw data:
City A: 6.9B revenue
City B: 6.8B revenue

The system produces:

🔻 Osaka underperforms due to lower ROI
🔺 Jakarta shows strong efficiency in promo spend
👉 Reduce promo in Osaka
👉 Increase investment in Jakarta

---

## 🏆 Highlights

- Built an **end-to-end AI data system locally**
- Designed a **semantic join engine** (auto table relationships)
- Implemented **multi-layer reasoning**:
  - Insight → Root Cause → Recommendation
- Optimized performance with caching & query strategy

---

## 🚀 Future Improvements

- Multi-user support
- Async LLM execution
- KPI alert system
- API deployment (FastAPI)
- Cloud scaling (AWS/GCP)

---

## 👤 Author

**Beny Susanto**

- 💼 Data Analyst / Data Engineer mindset
- 📊 Focus: turning data into business decisions

---