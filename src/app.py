"""
app.py — Complaint Auto-Routing System
  • CLI mode  : python app.py --mode cli
  • Web mode  : python app.py --mode web   (default)

Web app runs on http://localhost:5000 (Flask, no JS frameworks needed).
"""

import argparse, json, os, sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(__file__))

# ── CLI ────────────────────────────────────────────────────────────────────────

def run_cli():
    from inference import predict, predict_from_audio, predict_from_video

    print("\n" + "="*60)
    print("  Complaint Auto-Routing System  [CLI Mode]")
    print("="*60)
    print("Input types: 1) Text   2) Audio (.wav)   3) Video (.mp4)")
    choice = input("Select input type [1/2/3]: ").strip()

    if choice == "1":
        text = input("Enter complaint text:\n> ").strip()
        result = predict(text)
    elif choice == "2":
        path = input("Audio file path (.wav): ").strip()
        result = predict_from_audio(path)
        print(f"\nTranscribed: {result.get('transcribed_text')}")
    elif choice == "3":
        path = input("Video file path (.mp4): ").strip()
        result = predict_from_video(path)
        print(f"\nTranscribed: {result.get('transcribed_text')}")
    else:
        print("Invalid choice."); return

    _print_result(result)

def _print_result(r):
    print("\n" + "="*60)
    print("  ANALYSIS RESULT")
    print("="*60)
    print(f"  Processed Text : {r['processed_text'][:80]}…")
    print(f"\n  📋 Officer Assigned")
    o = r["officer"]
    print(f"     Name       : {o['name']}")
    print(f"     Department : {o['department']}")
    print(f"     Confidence : {o['confidence']}")
    p = r["priority"]
    print(f"\n  🚦 Priority    : {p['level']}  ({p['confidence']})")
    print(f"  ⏱  ETA         : {r['eta_days']} day(s)")
    print(f"\n  🔍 Similar Past Complaints")
    for i, s in enumerate(r["similar_complaints"], 1):
        print(f"     {i}. [{s['id']}] {s['text']}")
        print(f"        Priority: {s['priority']} | ETA: {s['eta_days']}d | Score: {s['score']}")
    print("="*60 + "\n")

# ── Web App (Flask) ────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Complaint Auto-Routing System</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,sans-serif;background:#f0f4f8;color:#1a202c;min-height:100vh}
  header{background:#1a365d;color:#fff;padding:20px 40px}
  header h1{font-size:1.4rem;letter-spacing:.5px}
  header p{font-size:.85rem;opacity:.7;margin-top:4px}
  main{max-width:900px;margin:30px auto;padding:0 20px}
  .card{background:#fff;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,.08);padding:28px;margin-bottom:24px}
  h2{font-size:1rem;font-weight:600;margin-bottom:14px;color:#2d3748}
  textarea{width:100%;min-height:100px;border:1px solid #cbd5e0;border-radius:8px;padding:10px;
    font-size:.93rem;resize:vertical;outline:none}
  textarea:focus{border-color:#3182ce;box-shadow:0 0 0 3px rgba(49,130,206,.15)}
  .upload-row{display:flex;gap:12px;margin-top:12px;flex-wrap:wrap}
  .upload-label{display:flex;align-items:center;gap:6px;padding:8px 14px;border:1px dashed #a0aec0;
    border-radius:8px;cursor:pointer;font-size:.85rem;color:#4a5568;background:#f7fafc}
  .upload-label:hover{border-color:#3182ce;color:#3182ce}
  input[type=file]{display:none}
  button{background:#1a365d;color:#fff;border:none;border-radius:8px;padding:10px 28px;
    font-size:.93rem;cursor:pointer;margin-top:14px;transition:background .2s}
  button:hover{background:#2a4a7f}
  #result{display:none}
  .grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:4px}
  .stat{background:#edf2f7;border-radius:8px;padding:14px;text-align:center}
  .stat .val{font-size:1.5rem;font-weight:700;color:#1a365d}
  .stat .lbl{font-size:.75rem;color:#718096;margin-top:4px;text-transform:uppercase;letter-spacing:.5px}
  .badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.8rem;font-weight:600}
  .High{background:#fed7d7;color:#c53030}
  .Medium{background:#fefcbf;color:#744210}
  .Low{background:#c6f6d5;color:#276749}
  table{width:100%;border-collapse:collapse;font-size:.85rem;margin-top:8px}
  th{background:#edf2f7;text-align:left;padding:8px;font-weight:600;color:#4a5568}
  td{padding:8px;border-top:1px solid #e2e8f0;vertical-align:top}
  .score-bar{height:6px;background:#bee3f8;border-radius:3px;margin-top:4px}
  .score-fill{height:6px;background:#3182ce;border-radius:3px}
  #loading{display:none;text-align:center;padding:20px;color:#718096}
  .spinner{display:inline-block;width:24px;height:24px;border:3px solid #e2e8f0;
    border-top-color:#3182ce;border-radius:50%;animation:spin .8s linear infinite;margin-right:8px;vertical-align:middle}
  @keyframes spin{to{transform:rotate(360deg)}}
  .fname{font-size:.8rem;color:#3182ce;margin-top:4px}
</style>
</head>
<body>
<header>
  <h1>⚙ Complaint Auto-Routing System</h1>
  <p>AI-powered complaint processing — officer routing · priority · ETA · similarity</p>
</header>
<main>
  <div class="card">
    <h2>Submit Complaint</h2>
    <textarea id="complaintText" placeholder="Type your complaint here (any language)…"></textarea>
    <div class="upload-row">
      <label class="upload-label">
        🎙 Upload Audio (.wav)
        <input type="file" id="audioFile" accept=".wav,.mp3,.flac">
      </label>
      <label class="upload-label">
        🎬 Upload Video (.mp4)
        <input type="file" id="videoFile" accept=".mp4,.avi,.mov">
      </label>
    </div>
    <div id="audioName" class="fname"></div>
    <div id="videoName" class="fname"></div>
    <br>
    <button onclick="submitComplaint()">Analyse Complaint</button>
  </div>

  <div id="loading"><span class="spinner"></span>Analysing complaint…</div>

  <div id="result">
    <div class="card">
      <h2>📊 Prediction Summary</h2>
      <div class="grid">
        <div class="stat">
          <div class="val" id="res-officer">—</div>
          <div class="lbl">Assigned Officer</div>
          <div style="font-size:.75rem;color:#718096;margin-top:4px" id="res-dept"></div>
        </div>
        <div class="stat">
          <div class="val" id="res-priority">—</div>
          <div class="lbl">Priority Level</div>
        </div>
        <div class="stat">
          <div class="val" id="res-eta">—</div>
          <div class="lbl">ETA (days)</div>
        </div>
      </div>
      <p style="font-size:.8rem;color:#718096;margin-top:12px" id="res-conf"></p>
    </div>

    <div class="card">
      <h2>🔍 Similar Past Complaints</h2>
      <table>
        <thead><tr><th>ID</th><th>Complaint</th><th>Priority</th><th>ETA</th><th>Similarity</th></tr></thead>
        <tbody id="similar-tbody"></tbody>
      </table>
    </div>
  </div>
</main>

<script>
document.getElementById('audioFile').onchange = e => {
  document.getElementById('audioName').textContent = e.target.files[0]?.name || '';
};
document.getElementById('videoFile').onchange = e => {
  document.getElementById('videoName').textContent = e.target.files[0]?.name || '';
};

async function submitComplaint() {
  const text  = document.getElementById('complaintText').value.trim();
  const audio = document.getElementById('audioFile').files[0];
  const video = document.getElementById('videoFile').files[0];

  if (!text && !audio && !video) { alert('Please provide complaint text, audio, or video.'); return; }

  document.getElementById('loading').style.display = 'block';
  document.getElementById('result').style.display  = 'none';

  let response;
  if (audio) {
    const fd = new FormData(); fd.append('file', audio); fd.append('type', 'audio');
    response = await fetch('/predict', { method:'POST', body: fd });
  } else if (video) {
    const fd = new FormData(); fd.append('file', video); fd.append('type', 'video');
    response = await fetch('/predict', { method:'POST', body: fd });
  } else {
    response = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ text })
    });
  }

  const data = await response.json();
  document.getElementById('loading').style.display = 'none';

  if (data.error) { alert('Error: ' + data.error); return; }

  const p = data.priority;
  document.getElementById('res-officer').textContent  = data.officer.name;
  document.getElementById('res-dept').textContent     = data.officer.department;
  document.getElementById('res-priority').innerHTML   =
    `<span class="badge ${p.level}">${p.level}</span>`;
  document.getElementById('res-eta').textContent      = data.eta_days + 'd';
  document.getElementById('res-conf').textContent     =
    `Officer confidence: ${data.officer.confidence}  |  Priority confidence: ${p.confidence}`;

  const tbody = document.getElementById('similar-tbody');
  tbody.innerHTML = '';
  data.similar_complaints.forEach(s => {
    const pct = Math.round(parseFloat(s.score) * 100);
    tbody.innerHTML += `<tr>
      <td style="white-space:nowrap"><b>${s.id}</b></td>
      <td>${s.text}</td>
      <td><span class="badge ${s.priority}">${s.priority}</span></td>
      <td>${s.eta_days}d</td>
      <td>${pct}%<div class="score-bar"><div class="score-fill" style="width:${pct}%"></div></div></td>
    </tr>`;
  });

  document.getElementById('result').style.display = 'block';
}
</script>
</body>
</html>
"""

def run_web(host="0.0.0.0", port=5000):
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask not found. Install it: pip install flask")
        sys.exit(1)

    from inference import predict, predict_from_audio, predict_from_video
    import tempfile

    app = Flask(__name__)

    @app.route("/")
    def index():
        from flask import Response
        return Response(HTML, mimetype="text/html")

    @app.route("/predict", methods=["POST"])
    def api_predict():
        try:
            ct = request.content_type or ""
            if "multipart" in ct:
                file = request.files.get("file")
                ftype = request.form.get("type", "audio")
                suffix = ".wav" if ftype == "audio" else ".mp4"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    file.save(tmp.name)
                    if ftype == "audio":
                        result = predict_from_audio(tmp.name)
                    else:
                        result = predict_from_video(tmp.name)
                os.unlink(tmp.name)
            else:
                body = request.get_json(force=True)
                result = predict(body.get("text", ""))
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    print(f"\n  Complaint Auto-Routing System")
    print(f"  Web UI → http://localhost:{port}\n")
    app.run(host=host, port=port, debug=False)

# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cli", "web"], default="web")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    # make sure models exist
    model_dir = os.path.join(ROOT, "models")
    if not os.path.exists(os.path.join(model_dir, "tfidf.pkl")):
        print("Models not found. Running training first …\n")
        import subprocess
        subprocess.run([sys.executable, os.path.join(ROOT, "src", "train.py")], check=True)

    if args.mode == "cli":
        run_cli()
    else:
        run_web(port=args.port)
