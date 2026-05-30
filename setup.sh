#!/usr/bin/env bash
# setup.sh — Install dependencies, generate data, train models, launch app.
set -e

echo "========================================"
echo "  Complaint Auto-Routing System Setup"
echo "========================================"

# 1. Install Python deps
echo "[1/4] Installing Python dependencies..."
pip install flask scikit-learn numpy --quiet

# Optional multimodal deps (comment out if not needed)
pip install SpeechRecognition moviepy deep-translator --quiet 2>/dev/null || true

# 2. Generate synthetic training data
echo "[2/4] Generating training data..."
python data/generate_data.py

# 3. Train models
echo "[3/4] Training ML models..."
python src/train.py

# 4. Run evaluation
echo "[4/4] Evaluation metrics..."
python src/evaluate.py

echo ""
echo "========================================"
echo "  Setup complete!"
echo "  Run the web app : python src/app.py"
echo "  Run CLI mode    : python src/app.py --mode cli"
echo "========================================"
