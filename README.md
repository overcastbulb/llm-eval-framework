# 🧪 LLM Evaluation Framework

A modular framework for evaluating Large Language Models across four key dimensions: **Accuracy**, **Hallucination**, **Relevance**, and **Coherence** — with a Streamlit dashboard for visualizing results.

---

## 🧠 How It Works

```
Test Cases (JSON) → Groq LLM → Generate Answers
                                      ↓
                    ┌─────────────────────────────┐
                    │  Accuracy Evaluator          │  (semantic + exact match)
                    │  Hallucination Evaluator     │  (LLM-as-judge)
                    │  Relevance Evaluator         │  (LLM-as-judge)
                    │  Coherence Evaluator         │  (LLM-as-judge)
                    └─────────────────────────────┘
                                      ↓
                         Results (JSON) → Streamlit Dashboard
```

---

## 📁 Project Structure

```
llm-eval-framework/
├── data/
│   └── test_cases.json         # Q&A evaluation dataset
├── evaluators/
│   ├── accuracy.py             # Semantic + exact match scoring
│   ├── hallucination.py        # Groundedness check via LLM-as-judge
│   ├── relevance.py            # Answer relevance scoring
│   └── coherence.py            # Writing clarity scoring
├── llm/
│   └── groq_client.py          # Groq API wrapper
├── results/                    # Auto-saved JSON results
├── eval_runner.py              # Main evaluation pipeline
├── dashboard.py                # Streamlit visual dashboard
├── requirements.txt
└── .env.example
```

---

## ⚙️ Setup

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/llm-eval-framework.git
cd llm-eval-framework
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your Groq API key
```bash
export GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 3. Run evaluation
```bash
python3 eval_runner.py
```

### 4. Launch dashboard
```bash
streamlit run dashboard.py
```

---

## 📊 Evaluation Metrics

| Metric | Method | Description |
|---|---|---|
| **Accuracy** | Semantic similarity (BGE embeddings) + exact match | How correct is the answer? |
| **Hallucination** | LLM-as-judge (Groq) | Is the answer grounded in context? |
| **Relevance** | LLM-as-judge (Groq) | Does the answer address the question? |
| **Coherence** | LLM-as-judge (Groq) | Is the answer clear and well-structured? |

All scores range from **0.0** (worst) to **1.0** (best).

---

## 🖥 Dashboard Preview

- Summary score cards for all 4 metrics
- Bar chart comparing scores per question
- Radar chart of average performance
- Per-question drill-down with LLM reasoning

---

## 📦 Tech Stack

| Component | Tool |
|---|---|
| LLM | Llama3-8b via Groq API |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Dashboard | Streamlit + Plotly |
| Data | JSON test cases |

---

## 📄 License

MIT License