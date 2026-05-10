from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import joblib
import json
import os
import time
import urllib.parse
import html
import torch
import numpy as np
import queue
from collections import OrderedDict
from datetime import datetime
from transformers import BertTokenizer, BertModel
from scipy.sparse import csr_matrix, hstack
import hashlib

app = Flask(__name__)
CORS(app)

# ====================== NOTIFICATION SYSTEM ======================
listeners = []

# ====================== KONFIGURASI ======================
MODEL_PATH      = "xss_hybrid_model.pkl"
TFIDF_PATH      = "tfidf_vectorizer.pkl"
BERT_MODEL_NAME = "bert-base-uncased"

# ====================== LOGGING & STATS RIL-TIME ======================
LOGS_HISTORY = []
MAX_LOGS = 100
LIVE_STATS = {
    "total_requests": 0,
    "xss_detected": 0,
    "safe_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0
}

# ====================== EMBEDDING CACHE ======================
EMBEDDING_CACHE = OrderedDict()
MAX_CACHE_SIZE = 10000

# ====================== PREPROCESSING ======================
def preprocess_payload(text: str) -> str:
    """
    Preprocessing identik dengan yang digunakan saat training:
    1. URL decoding
    2. HTML entity decoding
    3. Normalisasi whitespace
    """
    try:
        text = urllib.parse.unquote(str(text))
        text = html.unescape(text)
        text = " ".join(text.split())
    except Exception:
        pass
    return text

# ====================== LOAD MODEL ======================
MODEL_LOADED = False
MODEL_ERROR  = None
device       = None

try:
    print("Memuat model dan komponen BERT (harap tunggu)...")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True

    lr_model         = joblib.load(MODEL_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_PATH)

    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model     = BertModel.from_pretrained(BERT_MODEL_NAME)
    bert_model.eval()

    # Try GPU first, fallback to CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("✅ CUDA detected, using GPU")
    else:
        device = torch.device('cpu')
        print("📌 CUDA not available, using CPU")
    
    bert_model.to(device)

    MODEL_LOADED = True
    print(f"✅ Model berhasil dimuat | Device: {device}")
except Exception as e:
    MODEL_ERROR = str(e)
    print(f"❌ Gagal memuat model: {e}")

# ====================== LOAD JSON ======================
METRICS_STATIC = {}
DATASET_INFO   = {}

if os.path.exists("metrics.json"):
    with open("metrics.json", "r", encoding="utf-8") as f:
        METRICS_STATIC = json.load(f)

if os.path.exists("dataset_info.json"):
    with open("dataset_info.json", "r", encoding="utf-8") as f:
        DATASET_INFO = json.load(f)

# ====================== HELPER ======================
def get_text_hash(text: str) -> str:
    """Generate hash untuk text caching"""
    return hashlib.md5(text.encode()).hexdigest()

def get_bert_embeddings_cached(text: str) -> tuple[np.ndarray, bool]:
    """Ekstraksi BERT CLS token embedding dengan caching"""
    text_hash = get_text_hash(text)

    if text_hash in EMBEDDING_CACHE:
        LIVE_STATS["cache_hits"] += 1
        EMBEDDING_CACHE.move_to_end(text_hash)
        return EMBEDDING_CACHE[text_hash], True

    LIVE_STATS["cache_misses"] += 1
    inputs = bert_tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    if len(EMBEDDING_CACHE) >= MAX_CACHE_SIZE:
        EMBEDDING_CACHE.popitem(last=False)
    EMBEDDING_CACHE[text_hash] = cls_embedding

    return cls_embedding, False

def build_hybrid_vector(text: str):
    """ Membangun hybrid feature vector dengan cached BERT embeddings """
    vec_tfidf = tfidf_vectorizer.transform([text])
    bert_embedding, _ = get_bert_embeddings_cached(text)
    vec_bert  = csr_matrix(bert_embedding)
    return hstack([vec_tfidf, vec_bert])

# ====================== ROUTES ======================

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name":        "XSS Detection API – Hybrid TF-IDF + BERT (OPTIMIZED)",
        "version":     "2.0.0",
        "description": "REST API untuk deteksi serangan XSS dengan fitur caching dan optimasi.",
        "endpoints": {
            "GET  /":        "Dokumentasi API",
            "GET  /health":  "Status model",
            "POST /predict": "Prediksi payload XSS",
            "GET  /metrics": "Metrik performa & Statistik Ril-time",
            "GET  /logs":    "Riwayat prediksi terbaru",
            "GET  /info":    "Informasi dataset",
            "GET  /cache-stats": "Cache statistics"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "device":       str(device) if MODEL_LOADED else None,
        "timestamp":    datetime.now().isoformat()
    }), 200 if MODEL_LOADED else 503

@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model tidak tersedia"}), 503

    data = request.get_json()
    raw_text = data.get("text", "")
    if not raw_text:
        return jsonify({"error": "Input kosong"}), 400

    try:
        # ===== TIMING TRACKING =====
        timings = {}
        
        # 1. Preprocessing
        t_start = time.time()
        processed_text = preprocess_payload(raw_text)
        timings["preprocessing"] = (time.time() - t_start) * 1000  # Convert to ms
        
        # 2. TF-IDF vectorization
        t_start = time.time()
        vec_tfidf = tfidf_vectorizer.transform([processed_text])
        timings["tfidf"] = (time.time() - t_start) * 1000
        
        # 3. BERT inference (with cache)
        t_start = time.time()
        bert_embedding, cache_hit = get_bert_embeddings_cached(processed_text)
        vec_bert = csr_matrix(bert_embedding)
        timings["bert"] = (time.time() - t_start) * 1000
        
        # 4. Hybrid vector + LR predict
        t_start = time.time()
        X_hybrid = hstack([vec_tfidf, vec_bert])
        raw_pred = lr_model.predict(X_hybrid)[0]
        is_xss = int(raw_pred) == 1
        label = "XSS" if is_xss else "Benign"
        timings["prediction"] = (time.time() - t_start) * 1000

        # Confidence
        confidence = 0.0
        try:
            proba = lr_model.predict_proba(X_hybrid)[0]
            confidence = float(max(proba))
        except: pass

        # Total latency
        timings["total"] = sum(timings.values())

        # UPDATE LIVE STATS
        LIVE_STATS["total_requests"] += 1
        if is_xss: LIVE_STATS["xss_detected"] += 1
        else: LIVE_STATS["safe_requests"] += 1

        # LOG KE HISTORY
        log_entry = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "query": raw_text,
            "processed_query": processed_text,
            "prediction": label,
            "confidence_score": round(confidence, 4),
            "is_safe": not is_xss,
            "timings": {k: round(v, 3) for k, v in timings.items()},
            "cache_hit": cache_hit
        }
        LOGS_HISTORY.insert(0, log_entry)
        if len(LOGS_HISTORY) > MAX_LOGS:
            LOGS_HISTORY.pop()

        cache_status = "💾 CACHE HIT" if log_entry["cache_hit"] else "🆕 NEW"
        print(f"[PREDICT] {'⚠️' if is_xss else '✅'} {label} | {cache_status} | Total: {timings['total']:.2f}ms | Input: {raw_text[:50]}")
        
        # NOTIFIKASI RIL-TIME via SSE
        for q in listeners:
            try:
                q.put_nowait({"event": "new_log", "is_safe": not is_xss})
            except: pass

        return jsonify(log_entry), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    # Gabungkan metrics skripsi (static) dengan statistik ril-time (dynamic)
    response = {**METRICS_STATIC, **LIVE_STATS}
    response["cache_hit_rate"] = round(
        LIVE_STATS["cache_hits"] / max(1, LIVE_STATS["total_requests"]) * 100, 2
    )
    return jsonify(response), 200

@app.route("/cache-stats", methods=["GET"])
def cache_stats():
    return jsonify({
        "cache_size": len(EMBEDDING_CACHE),
        "max_cache_size": MAX_CACHE_SIZE,
        "cache_hits": LIVE_STATS["cache_hits"],
        "cache_misses": LIVE_STATS["cache_misses"],
        "hit_rate": round(
            LIVE_STATS["cache_hits"] / max(1, LIVE_STATS["cache_hits"] + LIVE_STATS["cache_misses"]) * 100, 2
        ) if (LIVE_STATS["cache_hits"] + LIVE_STATS["cache_misses"]) > 0 else 0,
        "total_requests": LIVE_STATS["total_requests"]
    }), 200

@app.route("/logs", methods=["GET"])
def get_logs():
    return jsonify(LOGS_HISTORY), 200

@app.route("/info", methods=["GET"])
def info():
    return jsonify(DATASET_INFO), 200

@app.route("/events", methods=["GET"])
def events():
    def generate():
        q = queue.Queue()
        listeners.append(q)
        try:
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield f"data: {json.dumps(msg)}\n\n"
                except queue.Empty:
                    yield f": {datetime.now().isoformat()}\n\n"
        finally:
            listeners.remove(q)
    
    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)