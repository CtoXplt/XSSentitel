from collections import OrderedDict
import hashlib
import html
import json
import os
import queue
import time
import urllib.parse
from datetime import datetime
import logging

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from scipy.sparse import csr_matrix, hstack
from transformers import BertModel, BertTokenizer

# NOTE: This API is implemented using FastAPI (ASGI), not Flask.
# Use an ASGI server like Uvicorn or Gunicorn+UvicornWorker for production.

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xss-api")

app = FastAPI(
    title="XSS Detection API – Hybrid TF-IDF + BERT (FastAPI)",
    version="2.1.0",
    description="REST API untuk deteksi serangan XSS dengan FastAPI, cache BERT embeddings, dan metrik realtime.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan whitelist domain frontend jika sudah tahu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== KONFIGURASI ======================
MODEL_PATH = "xss_hybrid_model.pkl"
TFIDF_PATH = "tfidf_vectorizer.pkl"
BERT_MODEL_NAME = "bert-base-uncased"

# ====================== LOGGING & STATISTICS ======================
listeners = []
LOGS_HISTORY = []
MAX_LOGS = 100
LIVE_STATS = {
    "total_requests": 0,
    "xss_detected": 0,
    "safe_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
}

# ====================== EMBEDDING CACHE ======================
EMBEDDING_CACHE = OrderedDict()
MAX_CACHE_SIZE = 10000

# ====================== PREPROCESSING ======================
def preprocess_payload(text: str) -> str:
    try:
        text = urllib.parse.unquote(str(text))
        text = html.unescape(text)
        text = " ".join(text.split())
    except Exception:
        pass
    return text

# ====================== MODEL LOADING ======================
MODEL_LOADED = False
MODEL_ERROR = None
device = None

try:
    logger.info("Memuat model dan komponen BERT (harap tunggu)...")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True

    lr_model = joblib.load(MODEL_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_PATH)

    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)
    bert_model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("✅ CUDA detected, using GPU")
    else:
        device = torch.device("cpu")
        logger.info("📌 CUDA not available, using CPU")

    bert_model.to(device)
    MODEL_LOADED = True
    logger.info(f"✅ Model berhasil dimuat | Device: {device}")
except Exception as e:
    MODEL_ERROR = str(e)
    logger.error(f"❌ Gagal memuat model: {e}")

# ====================== LOAD JSON ======================
METRICS_STATIC = {}
DATASET_INFO = {}

if os.path.exists("metrics.json"):
    with open("metrics.json", "r", encoding="utf-8") as f:
        METRICS_STATIC = json.load(f)

if os.path.exists("dataset_info.json"):
    with open("dataset_info.json", "r", encoding="utf-8") as f:
        DATASET_INFO = json.load(f)

# ====================== HELPERS ======================
def get_text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def get_bert_embeddings_cached(text: str) -> tuple[np.ndarray, bool]:
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
    vec_tfidf = tfidf_vectorizer.transform([text])
    bert_embedding, _ = get_bert_embeddings_cached(text)
    vec_bert = csr_matrix(bert_embedding)
    return hstack([vec_tfidf, vec_bert])

# ====================== Pydantic Model ======================
class PredictRequest(BaseModel):
    text: str

# ====================== ROUTES ======================
@app.get("/")
def index():
    return JSONResponse({
        "name": "XSS Detection API – Hybrid TF-IDF + BERT (FastAPI)",
        "version": "2.1.0",
        "description": "REST API untuk deteksi serangan XSS dengan FastAPI, cache BERT embeddings, dan metrik realtime.",
        "endpoints": {
            "GET /": "Dokumentasi API",
            "GET /health": "Status model",
            "POST /predict": "Prediksi payload XSS",
            "GET /metrics": "Metrik performa & Statistik Ril-time",
            "GET /logs": "Riwayat prediksi terbaru",
            "GET /info": "Informasi dataset",
            "GET /cache-stats": "Cache statistics",
            "POST /cache-clear": "Bersihkan cache embedding BERT"
        },
    })

@app.get("/health")
def health():
    return JSONResponse({
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "device": str(device) if MODEL_LOADED else None,
        "timestamp": datetime.now().isoformat(),
        "model_error": MODEL_ERROR if not MODEL_LOADED else None
    })

@app.post("/predict")
def predict(payload: PredictRequest):
    if not MODEL_LOADED:
        logger.error("Model tidak tersedia saat /predict dipanggil")
        raise HTTPException(status_code=503, detail="Model tidak tersedia")

    raw_text = payload.text
    if not raw_text:
        raise HTTPException(status_code=400, detail="Input kosong")

    try:
        timings = {}
        t_start = time.time()
        processed_text = preprocess_payload(raw_text)
        timings["preprocessing"] = (time.time() - t_start) * 1000

        t_start = time.time()
        vec_tfidf = tfidf_vectorizer.transform([processed_text])
        timings["tfidf"] = (time.time() - t_start) * 1000

        t_start = time.time()
        bert_embedding, cache_hit = get_bert_embeddings_cached(processed_text)
        vec_bert = csr_matrix(bert_embedding)
        timings["bert"] = (time.time() - t_start) * 1000

        t_start = time.time()
        X_hybrid = hstack([vec_tfidf, vec_bert])
        raw_pred = lr_model.predict(X_hybrid)[0]
        is_xss = int(raw_pred) == 1
        label = "XSS" if is_xss else "Benign"
        timings["prediction"] = (time.time() - t_start) * 1000

        confidence = 0.0
        try:
            proba = lr_model.predict_proba(X_hybrid)[0]
            confidence = float(max(proba))
        except Exception:
            pass

        timings["total"] = sum(timings.values())

        LIVE_STATS["total_requests"] += 1
        if is_xss:
            LIVE_STATS["xss_detected"] += 1
        else:
            LIVE_STATS["safe_requests"] += 1

        log_entry = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "query": raw_text,
            "processed_query": processed_text,
            "prediction": label,
            "confidence_score": round(confidence, 4),
            "is_safe": not is_xss,
            "timings": {k: round(v, 3) for k, v in timings.items()},
            "cache_hit": cache_hit,
        }
        LOGS_HISTORY.insert(0, log_entry)
        if len(LOGS_HISTORY) > MAX_LOGS:
            LOGS_HISTORY.pop()

        for q in listeners:
            try:
                q.put_nowait({"event": "new_log", "is_safe": not is_xss})
            except Exception:
                pass

        return JSONResponse(log_entry)
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    response = {**METRICS_STATIC, **LIVE_STATS}
    response["cache_hit_rate"] = round(
        LIVE_STATS["cache_hits"] / max(1, LIVE_STATS["total_requests"]) * 100, 2
    )
    return JSONResponse(response)

@app.get("/cache-stats")
def cache_stats():
    total = LIVE_STATS["cache_hits"] + LIVE_STATS["cache_misses"]
    return JSONResponse({
        "cache_size": len(EMBEDDING_CACHE),
        "max_cache_size": MAX_CACHE_SIZE,
        "cache_hits": LIVE_STATS["cache_hits"],
        "cache_misses": LIVE_STATS["cache_misses"],
        "hit_rate": round(
            LIVE_STATS["cache_hits"] / max(1, total) * 100, 2
        ) if total > 0 else 0,
        "total_requests": LIVE_STATS["total_requests"],
    })

@app.post("/cache-clear")
def cache_clear():
    EMBEDDING_CACHE.clear()
    LIVE_STATS["cache_hits"] = 0
    LIVE_STATS["cache_misses"] = 0
    return JSONResponse({"message": "Embedding cache cleared", "cache_size": len(EMBEDDING_CACHE)})

@app.get("/logs")
def get_logs():
    return JSONResponse(LOGS_HISTORY, status_code=200)

@app.get("/info")
def info():
    return JSONResponse(DATASET_INFO, status_code=200)

@app.get("/events")
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
            if q in listeners:
                listeners.remove(q)
    return StreamingResponse(generate(), media_type="text/event-stream")

# ========== CATATAN DEPLOY ==========
# Jalankan dengan: uvicorn api_fastapi:app --host 0.0.0.0 --port 5000 --workers 2
# Untuk production, gunakan reverse proxy (nginx/caddy) dan monitoring (Prometheus, Sentry, dsb)
