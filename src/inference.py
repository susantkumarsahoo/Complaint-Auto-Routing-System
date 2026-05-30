"""
inference.py — Load trained models and run predictions on new complaints.

Handles:
  - Text input (direct)
  - Audio input  (transcribed via SpeechRecognition + pocketsphinx / google offline)
  - Video input  (audio extracted via moviepy, then transcribed)
  - Multilingual (auto-detect + translate via deep-translator, offline-first)

All heavy imports are lazy so the CLI stays fast for text-only use.
"""

import json, os, pickle, warnings
import numpy as np

warnings.filterwarnings("ignore")

ROOT      = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(ROOT, "models")
DATA_DIR  = os.path.join(ROOT, "data")

# ── officer lookup ─────────────────────────────────────────────────────────────

def _load_officers():
    path = os.path.join(DATA_DIR, "officers.json")
    with open(path, encoding="utf-8") as f:
        return {o["id"]: o for o in json.load(f)}

OFFICERS = _load_officers()

# ── model loading (lazy, cached) ───────────────────────────────────────────────

_cache = {}

def _load(name):
    if name not in _cache:
        with open(os.path.join(MODEL_DIR, name), "rb") as f:
            _cache[name] = pickle.load(f)
    return _cache[name]

def _tfidf():    return _load("tfidf.pkl")
def _off_clf():  return _load("officer_clf.pkl")
def _off_le():   return _load("officer_le.pkl")
def _pri_clf():  return _load("priority_clf.pkl")
def _pri_le():   return _load("priority_le.pkl")
def _eta_reg():  return _load("eta_reg.pkl")
def _corpus():   return _load("corpus.pkl")

# ── multimodal transcription ───────────────────────────────────────────────────

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio using SpeechRecognition.
    Falls back through: sphinx (offline) → google (online if available).
    """
    try:
        import speech_recognition as sr
    except ImportError:
        raise ImportError("Install SpeechRecognition: pip install SpeechRecognition pocketsphinx")

    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)

    # try offline sphinx first
    try:
        return r.recognize_sphinx(audio)
    except Exception:
        pass
    # fallback: google (requires internet)
    try:
        return r.recognize_google(audio)
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")


def extract_audio_from_video(video_path: str) -> str:
    """Extract audio track from video, save as WAV, return path."""
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        raise ImportError("Install moviepy: pip install moviepy")

    tmp = video_path.rsplit(".", 1)[0] + "_audio.wav"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(tmp, verbose=False, logger=None)
    return tmp


def translate_to_english(text: str) -> str:
    """
    Translate non-English text to English using deep-translator (offline-first).
    If deep-translator not installed, returns text as-is.
    """
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated if translated else text
    except Exception:
        return text  # graceful degradation

# ── text normalisation ─────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    text = translate_to_english(text)
    return text.strip()

# ── similarity search ──────────────────────────────────────────────────────────

def find_similar(query_vec, top_k=3):
    """
    Cosine similarity between query and corpus.
    Returns list of (record, score) tuples.
    """
    corpus = _corpus()
    matrix  = corpus["matrix"]          # sparse (N, vocab)
    records = corpus["records"]

    # cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(query_vec, matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(records[i], float(scores[i])) for i in top_idx]


def recall_at_k(k=3):
    """
    Compute Recall@K on the corpus (leave-one-out style).
    A hit = same officer_id in top-K results.
    """
    corpus  = _corpus()
    matrix  = corpus["matrix"]
    records = corpus["records"]

    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(matrix, matrix)
    np.fill_diagonal(sims, -1)          # exclude self

    hits = 0
    for i, rec in enumerate(records):
        top = np.argsort(sims[i])[::-1][:k]
        if any(records[j]["officer_id"] == rec["officer_id"] for j in top):
            hits += 1
    return hits / len(records)

# ── main prediction ────────────────────────────────────────────────────────────

def predict(text: str) -> dict:
    clean = preprocess(text)
    vec   = _tfidf().transform([clean])

    # officer
    off_id   = _off_le().inverse_transform(_off_clf().predict(vec))[0]
    off_prob = float(_off_clf().predict_proba(vec).max())
    officer  = OFFICERS.get(off_id, {"id": off_id, "name": "Unknown", "department": "Unknown"})

    # priority
    pri_label = _pri_le().inverse_transform(_pri_clf().predict(vec))[0]
    pri_prob  = float(_pri_clf().predict_proba(vec).max())

    # eta
    eta = max(1, round(float(_eta_reg().predict(vec)[0])))

    # similar complaints
    similar = find_similar(vec, top_k=3)

    return {
        "processed_text": clean,
        "officer": {
            "id":         officer["id"],
            "name":       officer["name"],
            "department": officer["department"],
            "confidence": f"{off_prob:.1%}",
        },
        "priority": {
            "level":      pri_label,
            "confidence": f"{pri_prob:.1%}",
        },
        "eta_days": eta,
        "similar_complaints": [
            {
                "id":       rec["id"],
                "text":     rec["text"][:100] + ("…" if len(rec["text"]) > 100 else ""),
                "priority": rec["priority"],
                "eta_days": rec["eta_days"],
                "score":    f"{score:.3f}",
            }
            for rec, score in similar
        ],
    }


def predict_from_audio(audio_path: str) -> dict:
    text = transcribe_audio(audio_path)
    result = predict(text)
    result["transcribed_text"] = text
    return result


def predict_from_video(video_path: str) -> dict:
    audio_path = extract_audio_from_video(video_path)
    return predict_from_audio(audio_path)
