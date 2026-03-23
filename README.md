# 📊 AI Decision Intelligence System

An end-to-end AI-powered analytics system that converts natural language into SQL queries, analyzes data, detects anomalies, and provides business recommendations.

---

## 🚀 Features

- 🔍 Natural Language → SQL (LLM-powered)
- 📊 Automatic Data Visualization
- 💡 AI-generated Business Insights
- 🚨 Anomaly Detection (Z-score based)
- 🔍 Root Cause Analysis
- 🚀 Recommendation Engine (scoring-based)

---

## 🧠 How It Works

1. User asks a business question  
2. LLM converts question → SQL  
3. System executes query  
4. Generates:
   - Charts
   - Insights
   - Anomalies
   - Root cause
   - Recommendations  

---

## 🛠 Tech Stack

- Python
- FastAPI
- Streamlit
- Pandas
- LLM (DeepSeek / Local model)

---

## 📸 Example Queries

- "Show sales by product over time"
- "Compare sales across cities"
- "Top selling products"
- "Identify anomalies in revenue"

---

## ▶️ Run Locally

### 1. Start API

python -m uvicorn api.main:app --reload

### 1. Start UI
streamlit run streamlit_app.py