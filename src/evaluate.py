"""
evaluate.py — Report all evaluation metrics for the trained models.

Metrics:
  - Officer Routing   : Accuracy, F1-score (macro)
  - Priority Classif. : Accuracy, F1-score (macro), per-class report
  - ETA Regression    : MAE, RMSE
  - Similarity Search : Recall@1, Recall@3, Recall@5

Run: python evaluate.py
"""

import os, sys, pickle, json
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.metrics import (accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = os.path.join(ROOT, "models")

def load(name):
    with open(os.path.join(MODEL_DIR, name), "rb") as f:
        return pickle.load(f)

def main():
    print("\n" + "="*60)
    print("  EVALUATION REPORT — Complaint Auto-Routing System")
    print("="*60)

    tfidf   = load("tfidf.pkl")
    corpus  = load("corpus.pkl")
    records = corpus["records"]
    texts   = [r["text"] for r in records]
    X       = tfidf.transform(texts)

    # ── Officer Routing ────────────────────────────────────────────────────────
    clf_off = load("officer_clf.pkl")
    le_off  = load("officer_le.pkl")
    y_off   = le_off.transform([r["officer_id"] for r in records])
    _, X_te, _, y_te = train_test_split(X, y_off, test_size=0.2, random_state=42, stratify=y_off)
    y_pred = clf_off.predict(X_te)
    print(f"\n[1] Officer Routing")
    print(f"    Accuracy  : {accuracy_score(y_te, y_pred):.4f}")
    print(f"    F1 (macro): {f1_score(y_te, y_pred, average='macro'):.4f}")

    # ── Priority Classification ────────────────────────────────────────────────
    clf_pri = load("priority_clf.pkl")
    le_pri  = load("priority_le.pkl")
    y_pri   = le_pri.transform([r["priority"] for r in records])
    _, X_te2, _, y_te2 = train_test_split(X, y_pri, test_size=0.2, random_state=42, stratify=y_pri)
    y_pred2 = clf_pri.predict(X_te2)
    print(f"\n[2] Priority Classification")
    print(f"    Accuracy  : {accuracy_score(y_te2, y_pred2):.4f}")
    print(f"    F1 (macro): {f1_score(y_te2, y_pred2, average='macro'):.4f}")
    print(classification_report(y_te2, y_pred2, target_names=le_pri.classes_, digits=3))

    # ── ETA Regression ────────────────────────────────────────────────────────
    reg_eta = load("eta_reg.pkl")
    y_eta   = np.array([r["eta_days"] for r in records], dtype=float)
    _, X_te3, _, y_te3 = train_test_split(X, y_eta, test_size=0.2, random_state=42)
    y_pred3 = reg_eta.predict(X_te3)
    mae  = mean_absolute_error(y_te3, y_pred3)
    rmse = mean_squared_error(y_te3, y_pred3) ** 0.5
    print(f"[3] ETA Regression")
    print(f"    MAE  : {mae:.4f} days")
    print(f"    RMSE : {rmse:.4f} days")

    # ── Similarity — Recall@K ──────────────────────────────────────────────────
    print(f"\n[4] Similarity Search (Recall@K, same officer as ground truth)")
    sims = cosine_similarity(corpus["matrix"], corpus["matrix"])
    np.fill_diagonal(sims, -1)
    for k in [1, 3, 5]:
        hits = 0
        for i, rec in enumerate(records):
            top = np.argsort(sims[i])[::-1][:k]
            if any(records[j]["officer_id"] == rec["officer_id"] for j in top):
                hits += 1
        print(f"    Recall@{k}: {hits/len(records):.4f}")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
