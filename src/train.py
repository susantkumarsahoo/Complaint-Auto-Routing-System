"""
train.py — Train all ML models for the complaint routing system.

Models trained:
  1. TF-IDF + Logistic Regression  → officer routing
  2. TF-IDF + Logistic Regression  → priority classification
  3. TF-IDF + Ridge Regression     → ETA prediction (days)
  4. TF-IDF matrix                 → saved for similarity search (cosine)

Run: python train.py
"""

import json, os, pickle, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, classification_report
)
from sklearn.preprocessing import LabelEncoder

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────

def load_data():
    path = os.path.join(DATA_DIR, "complaints.json")
    if not os.path.exists(path):
        print("Dataset not found. Running data generator …")
        sys.path.insert(0, DATA_DIR)
        import generate_data
        os.chdir(DATA_DIR)
        generate_data.generate_dataset  # already saved on import __main__
        import subprocess, sys as _sys
        subprocess.run([_sys.executable, os.path.join(DATA_DIR, "generate_data.py")], check=True)
        os.chdir(os.path.dirname(__file__))
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save(obj, name):
    with open(os.path.join(MODEL_DIR, name), "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved {name}")

# ── main ───────────────────────────────────────────────────────────────────────

def train():
    records = load_data()
    texts      = [r["text"]      for r in records]
    officer_ids = [r["officer_id"] for r in records]
    priorities  = [r["priority"]   for r in records]
    etas        = [r["eta_days"]   for r in records]

    print(f"\n{'='*55}")
    print(f"  Complaint Auto-Routing — Training Pipeline")
    print(f"  Samples: {len(texts)}")
    print(f"{'='*55}\n")

    # ── 1. TF-IDF vectoriser (shared) ─────────────────────────────────────────
    print("[1/4] Fitting TF-IDF vectoriser …")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X = tfidf.fit_transform(texts)
    save(tfidf, "tfidf.pkl")
    print(f"  Vocabulary size: {len(tfidf.vocabulary_)}")

    # ── 2. Officer routing ─────────────────────────────────────────────────────
    print("\n[2/4] Training Officer Routing model …")
    le_off = LabelEncoder()
    y_off  = le_off.fit_transform(officer_ids)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_off, test_size=0.2, random_state=42, stratify=y_off)
    clf_off = LogisticRegression(max_iter=1000, C=5, solver="lbfgs")
    clf_off.fit(X_tr, y_tr)
    y_pred = clf_off.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="macro")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  F1 (macro): {f1:.3f}")

    save(clf_off, "officer_clf.pkl")
    save(le_off,  "officer_le.pkl")

    # ── 3. Priority classification ─────────────────────────────────────────────
    print("\n[3/4] Training Priority Classification model …")
    le_pri = LabelEncoder()
    y_pri  = le_pri.fit_transform(priorities)

    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_pri, test_size=0.2, random_state=42, stratify=y_pri)
    clf_pri = LogisticRegression(max_iter=1000, C=3, solver="lbfgs")
    clf_pri.fit(X_tr2, y_tr2)
    y_pred2 = clf_pri.predict(X_te2)

    acc2 = accuracy_score(y_te2, y_pred2)
    f1_2 = f1_score(y_te2, y_pred2, average="macro")
    print(f"  Accuracy : {acc2:.3f}")
    print(f"  F1 (macro): {f1_2:.3f}")
    print(classification_report(y_te2, y_pred2, target_names=le_pri.classes_))

    save(clf_pri, "priority_clf.pkl")
    save(le_pri,  "priority_le.pkl")

    # ── 4. ETA regression ─────────────────────────────────────────────────────
    print("\n[4/4] Training ETA Regression model …")
    y_eta = np.array(etas, dtype=float)

    X_tr3, X_te3, y_tr3, y_te3 = train_test_split(X, y_eta, test_size=0.2, random_state=42)
    reg_eta = Ridge(alpha=1.0)
    reg_eta.fit(X_tr3, y_tr3)
    y_pred3 = reg_eta.predict(X_te3)
    mae = mean_absolute_error(y_te3, y_pred3)
    print(f"  MAE (days): {mae:.2f}")

    save(reg_eta, "eta_reg.pkl")

    # ── 5. Save full TF-IDF matrix + texts for similarity search ──────────────
    print("\nSaving corpus for similarity search …")
    save({"matrix": X, "records": records}, "corpus.pkl")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*55}")
    print(f"  Officer Routing   — Accuracy: {acc:.3f}  |  F1: {f1:.3f}")
    print(f"  Priority Classif. — Accuracy: {acc2:.3f}  |  F1: {f1_2:.3f}")
    print(f"  ETA Regression    — MAE     : {mae:.2f} days")
    print(f"{'='*55}\n")
    print("All models saved to /models/. Run python src/app.py to start.\n")

if __name__ == "__main__":
    train()
