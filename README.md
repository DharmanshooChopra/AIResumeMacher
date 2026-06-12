# 🛡️ AI Resume Macher: Contextual Resume Benchmarking 

AI Resume Macher is a production-ready, AI-driven resume parsing and matching engine designed to bridge the gap between job descriptions and professional profiles. It uses a sophisticated hybrid scoring model that combines keyword density with deep semantic understanding to provide a high-fidelity readiness assessment.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Sentence--Transformers-blueviolet?style=for-the-badge)
![Parser](https://img.shields.io/badge/Parser-Fault--Tolerant-green?style=for-the-badge)

## 🚀 Key Features

- **Hybrid Scoring Engine:** 
  - **70% Keyword Matching:** Uses section-aware weighting (Skills, Experience, Projects) to reward resumes that structure information correctly.
  - **30% Semantic Similarity:** Leverages the `multi-qa-mpnet-base-dot-v1` SBERT model for contextual understanding beyond simple keyword matches.
- **Fault-Tolerant PDF Extraction:** Dual-engine approach using `pdfplumber` for structured text and `pypdfium2` as a fallback for corrupted or non-standard document streams.
- **Intelligent Segmentation:** Automatically divides resumes into logical blocks (Experience, Education, Skills, etc.) for granular analysis.
- **Interactive Dashboard:** Built with Streamlit for a premium user experience featuring:
  - Match Quality Gauges
  - Detailed Skill Gap Profiles
  - Strategy advice for profile improvement
  - Document segment visualization

## 🛠️ Tech Stack

- **Frontend:** Streamlit (Custom CSS for Premium UI)
- **Natural Language Processing:** 
  - `spacy` (Text preprocessing)
  - `sentence-transformers` (Semantic encoding)
- **Document Parsing:** `pdfplumber`, `pypdfium2`
- **Data Handling:** `pandas`, `json`
- **Matching Algorithm:** Cosine Similarity + Weighted Keyword Mapping

## 📦 Installation

Ensure you have Python 3.10+ installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/[your-username]/Resume_Matcher_AI.git
   cd Resume_Matcher_AI
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download SpaCy Model:**
   The application will attempt to auto-download this on first run, but you can do it manually:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 🖥️ Usage

1. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

2. **Analyze a Resume:**
   - Select a **Benchmark Position** from the sidebar/dropdown.
   - Upload a **PDF Resume**.
   - Click **🚀 Execute Contextual Match Analysis**.

## 📁 Project Structure

```text
├── app.py              # Main Streamlit Application (UI & Orchestration)
├── utils.py            # AI Engine, Scoring Logic, and Text Processing
├── requirements.txt    # Project Dependencies
├── data/
│   └── jobs.json       # Structured job definitions (Benchmark Roles)
├── models/             # (Optional) Local model storage
└── assets/             # Images and UI assets
```

## ⚙️ Customization

You can add or modify job benchmarks by editing `data/jobs.json`. Each job follows this schema:

```json
{
  "job_id": "UNIQUE_ID",
  "job_title": "Role Title",
  "experience_required": "Experience Range",
  "job_summary": "Detailed JD summary for semantic matching",
  "required_skills": ["skill1", "skill2"],
  "keywords_for_matching": ["keyword1", "keyword2"]
}
```

---
© 2026 AI Resume Macher | Developed with Python 3.13 Ready Architecture.
