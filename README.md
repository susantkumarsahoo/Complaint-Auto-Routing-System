# Complaint Auto-Routing System

**Assessment Submission**

An offline, ML-driven complaint processing system that takes complaint text and automatically routes it to the right officer, predicts priority, estimates resolution time, and finds similar past complaints — all from complaint content alone, with no manual routing hints.

---

## What Is Actually Working (Honest Status)

| Feature | Status | Details |
|---|---|---|
| Text input |  Fully implemented & tested | Complete ML pipeline end to end |
| Officer routing |  Fully working | Logistic Regression, 8 departments |
| Priority prediction |  Fully working | High / Medium / Low with confidence |
| ETA prediction |  Fully working | Ridge Regression, output in days |
| Similarity search |  Fully working | Cosine similarity over TF-IDF corpus |
| Web UI |  Fully working | Flask app at localhost:5000 |
| CLI mode |  Fully working | `python src/app.py --mode cli` |
| Multilingual (English) |  Always works | No translation needed |
| Multilingual (other languages) |  Works with internet | Uses deep-translator → Google Translate → English → ML pipeline. Degrades gracefully if offline |
| Audio input |  Architecturally designed | Code written for SpeechRecognition + pocketsphinx pipeline. Not live-tested. Requires heavy ASR setup |
| Video input |  Architecturally designed | Code written for moviepy audio extraction + ASR. Not live-tested |

> **Note on audio/video:** The assignment states *"Text-only support can be treated as the minimum baseline, but the design should clearly account for multimodal extension."* The multimodal pipeline is fully designed and coded — the inference.py file shows exactly how audio and video would feed into the same ML pipeline via transcription. The architecture supports this extension without any model changes.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     INPUT LAYER                         │
│                                                         │
│   Text ──────────────────────────────────┐              │
│   Audio (.wav) → SpeechRecognition ──────┤              │
│   Video (.mp4) → moviepy → ASR ──────────┤              │
└──────────────────────────────────────────┼──────────────┘
                                           │ raw text
                                           ▼
┌─────────────────────────────────────────────────────────┐
│               MULTILINGUAL LAYER                        │
│   deep-translator (auto-detect → English)               │
│   Graceful fallback: if offline, pass text as-is        │
└──────────────────────────────────────────┬──────────────┘
                                           │ English text
                                           ▼
┌─────────────────────────────────────────────────────────┐
│               TF-IDF VECTORISER                         │
│   ngram_range=(1,2) | max_features=5000                 │
│   sublinear_tf=True | strip_accents=unicode             │
└──────┬───────────────────────────────────┬──────────────┘
       │ sparse feature vector             │
       ▼                                   ▼
┌──────────────────┐   ┌──────────────────────────────────┐
│  CLASSIFICATION  │   │         SIMILARITY SEARCH        │
│                  │   │                                  │
│  Officer Routing │   │  Cosine Similarity over          │
│  (LogReg, C=5)   │   │  TF-IDF corpus matrix            │
│                  │   │  → Top-K similar complaints      │
│  Priority Pred.  │   └──────────────────────────────────┘
│  (LogReg, C=3)   │
│                  │
│  ETA Regression  │
│  (Ridge, α=1.0)  │
└──────────────────┘
```

---

## ML Problem Framing

### Task 1 — Officer Routing (Multi-class Classification)
- **Input:** complaint text (TF-IDF vector)
- **Output:** one of 8 officer IDs
- **Model:** Logistic Regression with L2 regularisation
- **Why:** Sparse TF-IDF features work best with linear classifiers. LogReg gives calibrated probabilities (confidence scores) and is fully interpretable.

### Task 2 — Priority Prediction (Multi-class Classification)
- **Input:** complaint text (TF-IDF vector)
- **Output:** High / Medium / Low
- **Model:** Logistic Regression with L2 regularisation
- **Why:** Same reasoning. Three-class problem with clear textual signals (words like "sparking", "dangerous", "outbreak" → High).

### Task 3 — ETA Prediction (Regression)
- **Input:** complaint text (TF-IDF vector)
- **Output:** number of days (continuous)
- **Model:** Ridge Regression
- **Why:** Continuous target. Ridge handles multicollinearity in sparse features well. No overfitting risk even with 5000 features.

### Task 4 — Similarity Search (Retrieval)
- **Input:** TF-IDF vector of new complaint
- **Output:** Top-K most similar past complaints
- **Method:** Cosine similarity between query vector and full corpus matrix
- **Why:** Fully offline, no vector DB needed, effective for short civic complaint text. Scales to ~100K records before needing FAISS.

---

## Data Pipeline

### Synthetic Dataset
Since no labeled complaint dataset was available, a synthetic generator (`data/generate_data.py`) was built:

- **400 complaints** across 8 departments
- **40 base templates** × random augmentations (prefix/suffix variations)
- Labels: officer_id, priority (High/Medium/Low), eta_days
- Covers: Infrastructure, Sanitation, Electricity, Water Supply, Public Safety, Health, Transport, Environment

### Why synthetic data is acceptable here
The assessment evaluates **pipeline design and ML problem framing**, not dataset curation. The same pipeline runs identically on real labeled data — only the training CSV needs to be swapped.

---

## Project Structure

```
complaint_system/
├── data/
│   ├── generate_data.py     ← Synthetic dataset generator
│   ├── complaints.json      ← 400 labeled complaints (pre-generated)
│   ├── complaints.csv       ← Same data in CSV format
│   └── officers.json        ← Officer registry (8 officers)
├── models/                  ← Pre-trained .pkl files (ready to use)
│   ├── tfidf.pkl            ← Fitted TF-IDF vectoriser
│   ├── officer_clf.pkl      ← Officer routing classifier
│   ├── officer_le.pkl       ← Officer label encoder
│   ├── priority_clf.pkl     ← Priority classifier
│   ├── priority_le.pkl      ← Priority label encoder
│   ├── eta_reg.pkl          ← ETA regression model
│   └── corpus.pkl           ← Full TF-IDF matrix + records for similarity
├── src/
│   ├── train.py             ← Full ML training pipeline
│   ├── inference.py         ← Prediction engine (text + audio + video)
│   ├── evaluate.py          ← All evaluation metrics
│   └── app.py               ← Flask web app + CLI
├── requirements.txt
├── setup.sh                 ← One-shot setup script
└── README.md
```

---

## Quick Start

### Step 1 — Install dependencies

```bash
pip install flask scikit-learn numpy
```

Optional (for multilingual support):
```bash
pip install deep-translator
```

### Step 2 — Run (models already trained and included)

```bash
# Web app → http://localhost:5000
python src/app.py

# CLI mode
python src/app.py --mode cli
```

### Step 3 — Retrain from scratch (optional)

```bash
python data/generate_data.py   # regenerate dataset
python src/train.py            # retrain all models
```

### Step 4 — View evaluation metrics

```bash
python src/evaluate.py
```

Or one-shot everything:
```bash
bash setup.sh
```

---

## Evaluation Metrics

Measured on 20% held-out test set (80 samples, stratified split, random_state=42):

### Classification — Accuracy & F1

| Task | Accuracy | F1 (macro) |
|------|----------|------------|
| Officer Routing | **1.000** | **1.000** |
| Priority Classification | **1.000** | **1.000** |

Per-class breakdown (Priority):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| High | 1.00 | 1.00 | 1.00 | 49 |
| Medium | 1.00 | 1.00 | 1.00 | 24 |
| Low | 1.00 | 1.00 | 1.00 | 7 |

> Perfect scores are expected on synthetic data where test patterns closely match training templates. On real-world noisy data, expect Accuracy ~0.75–0.85 and F1 ~0.70–0.80.

### Regression — ETA Prediction

| Metric | Score |
|--------|-------|
| MAE | **1.20 days** |
| RMSE | **1.49 days** |

### Retrieval — Recall@K (Similarity Search)

| K | Recall@K |
|---|----------|
| 1 | **1.000** |
| 3 | **1.000** |
| 5 | **1.000** |

*Run `python src/evaluate.py` to reproduce all numbers.*

---

## Multilingual Support

```
English text        → directly into ML pipeline         offline
Hindi / Odia / etc  → deep-translator → English → ML    needs internet
Any language        → if offline, passes as-is to ML    accuracy drops
```

For a fully offline multilingual system, replace `deep-translator` with `argos-translate`:
```bash
pip install argos-translate
```
The pipeline architecture supports this swap with a single function change in `inference.py`.

---

## Officers & Departments

| ID | Officer Name | Department |
|----|-------------|------------|
| OFF001 | Rajesh Kumar | Infrastructure |
| OFF002 | Priya Sharma | Sanitation |
| OFF003 | Anil Patro | Electricity |
| OFF004 | Sneha Mishra | Water Supply |
| OFF005 | Deepak Nayak | Public Safety |
| OFF006 | Kavita Das | Health |
| OFF007 | Suresh Behera | Transport |
| OFF008 | Meena Rath | Environment |

---

## Trade-offs & Design Decisions

| Decision | What was chosen | Why | What could replace it |
|----------|----------------|-----|----------------------|
| Embeddings | TF-IDF | Zero-GPU, fully offline, fast | `sentence-transformers/all-MiniLM-L6-v2` for better semantic understanding |
| Classifier | Logistic Regression | Interpretable, fast, works well on sparse vectors | SVM, fine-tuned BERT for production |
| Similarity | In-memory cosine | No dependencies, works offline | FAISS for 100K+ complaint corpora |
| Translation | deep-translator | Easy pip install, auto language detect | `argos-translate` for fully offline multilingual |
| ASR | SpeechRecognition | Simple API, multiple backends | OpenAI Whisper (local) for production-grade offline transcription |
| Data | Synthetic | No labeled dataset available | Replace `complaints.json` with real data — zero pipeline changes needed |

---

## Production Extension Path

```
Current (Demo)              →    Production
─────────────────────────────────────────────
TF-IDF                      →    sentence-transformers
In-memory cosine            →    FAISS vector index
Synthetic data              →    Real labeled complaint DB
deep-translator             →    argos-translate (fully offline)
SpeechRecognition/sphinx    →    Whisper (local, high accuracy)
Flask dev server            →    Gunicorn + Nginx
pickle models               →    MLflow model registry
```

---

## Requirements

- Python 3.8+
- No GPU required
- No external paid APIs
- No rule-based routing logic — all decisions are model-driven

Core dependencies:
```
flask
scikit-learn
numpy
pandas
```

Optional:
```
deep-translator     # multilingual support
SpeechRecognition   # audio transcription
moviepy             # video audio extraction
```


## User Input Dart (example to test the app)

1.  Electricity — OFF003 Anil Patro

```
The transformer near our colony has been sparking for 2 days and there is no electricity at all. It is extremely dangerous and children are playing near it. Please send someone immediately.
```
2.  Sanitation — OFF002 Priya Sharma

```
Garbage has not been collected from our street for 12 days. The waste is overflowing onto the road, there is unbearable smell, and rats are visible everywhere. Many residents are falling sick.
```
3.  Water Supply — OFF004 Sneha Mishra
```
Water supply has been completely cut off in our entire sector for 4 days. The water that does come is brown and smells rotten. Several children and elderly people have fallen sick after drinking it.
```
4.  Infrastructure — OFF001 Rajesh Kumar
```
There is a massive pothole on the main road that has already caused 3 bike accidents this week. A manhole cover nearby is also missing. The road is completely unsafe especially at night.
```
5.  Public Safety — OFF005 Deepak Nayak
```
A pack of stray dogs has been attacking schoolchildren every morning near the gate for one week. Three children have been bitten and parents are afraid to send their kids to school.
```
6.  Health — OFF006 Kavita Das
```
There is a serious dengue outbreak in our ward with over 20 cases reported this week. Mosquitoes are breeding in stagnant water everywhere and the health department has taken no action at all.
```
7.  Transport — OFF007 Suresh Behera
```
Bus route 45 has been completely suspended for 3 weeks without any notice. Hundreds of daily commuters including students and office workers have no transport option available.
```
8.  Environment — OFF008 Meena Rath
```
An illegal factory is dumping chemical waste directly into the river every night. The water is turning black and residents near the river are experiencing severe breathing problems and skin rashes.
```
