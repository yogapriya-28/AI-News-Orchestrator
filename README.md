# 🧭 AI News Orchestrator — PRO
Reconstruct the real story from scattered news.

AI News Orchestrator PRO is an AI-powered news-intelligence system that:
- Aggregates news from multiple sources  
- Extracts key milestones  
- Builds a chronological timeline  
- Detects conflicting facts  
- Generates multi-language summaries  
- Visualizes timelines  
- Rates source credibility  

It transforms messy, contradictory news into a clean narrative of **how an event evolved**.

---

## 📘 Project Overview
Users enter any event/topic (e.g., “ISRO Aditya-L1”, “GPT-5 Launch”, “COP28”).  
The system automatically fetches news → extracts events → merges duplicates → builds a timeline → summarizes the story.

No dataset required — everything is fetched live.

---

## 🌟 Features

### 🔍 1. Multi-Source News Aggregation
Fetches news from:
- **NewsAPI** (primary)
- **Google News RSS** (free fallback)

With:
- Real URL extraction (Google redirect fix)
- Low-quality source filtering (e.g., IAS/GK portals)
- Metadata (title, URL, summary, date)

---

### 🧠 2. Event Extraction Engine
Uses NLP + rule-based logic:
- Sentence splitting
- Date detection (ISO, RSS, regex, dateparser)
- Event keywords (launch, announce, confirm, strike…)
- Extracts meaningful milestones

---

### ♻️ 3. Smart Deduplication
Merges duplicate/near-duplicate events using:
- Text normalization  
- Similarity scoring  
- Source merging  
- Longest/most descriptive event chosen as representative

Produces a clean, chronological timeline.

---

### 🕒 4. Vertical Timeline Visualization
Built using Plotly:
- Category-based colors  
- Icons (🚀 AI • 🗳️ Politics • 🔬 Science • 💼 Business)  
- Hover preview  
- Export as PNG (kaleido)

---

### 🌍 5. Multi-Language Summary
Supports:
- English (en)
- Hindi (hi)
- Tamil (ta)
- French (fr)
- Spanish (es)
- Arabic (ar)

Pipeline:
1. Detect language  
2. Translate (googletrans)  
3. Summarize (extractive or OpenAI)  

---

### 🧩 6. Advanced Analysis (PRO Mode)

#### ✔ Verified Facts (NER)
Extracts:
- Dates  
- Numbers  
- Locations  
- Organizations  
- People  

#### ✔ Conflict Detection
Finds contradictions:
- Number mismatches  
- Date mismatches  

#### ✔ Bias / Clickbait Detection
Scores:
- Clickbait intensity  
- Subjectivity  
- Sentiment  

#### ✔ Authenticity Score
Grades domains (A+ → D) using weighted trust scores.

---

## 📦 Export Options
The system generates:
- `timeline.csv`
- `timeline.json`
- `timeline.png`
- `advanced_report.json`

---

## 💾 Timeline History (SQLite)
Every timeline is saved automatically.  
Users can reload past analyses from the **History** tab.

---

## 🚀 Running the App

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-name>/AI-News-Orchestrator.git
cd AI-News-Orchestrator
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Install spaCy model
```bash
python -m spacy download en_core_web_sm
```

### 4️⃣ Run Streamlit
```bash
streamlit run news_orchestrator_competition_pro.py
```

---

## 🛠 Tech Stack

**Backend & NLP**
- Python 3.10+
- spaCy (NER)
- dateparser
- googletrans
- langdetect
- feedparser
- NLTK (VADER)

**Frontend**
- Streamlit
- Plotly (timeline)

**Optional AI**
- OpenAI API (GPT-4o mini / GPT-5 mini)

**Database**
- SQLite (timeline history)

---

## 📂 Project Structure
```
AI-News-Orchestrator/
│── news_orchestrator_competition_pro.py
│── requirements.txt
│── README.md
└── timeline_history.db   (auto-created)
```

---

## 🎥 Demo (Optional)
(Add screenshots / GIF later)

---

## 🤝 Contributing
Pull requests and suggestions are welcome.

---

## 📜 License
MIT License
