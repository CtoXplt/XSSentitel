import gradio as gr
import joblib
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from scipy.sparse import csr_matrix, hstack
import urllib.parse
import html
import time

# Load models and components
MODEL_PATH = "xss_hybrid_model.pkl"
TFIDF_PATH = "tfidf_vectorizer.pkl"
BERT_MODEL_NAME = "bert-base-uncased"

print("Loading models...")
lr_model = joblib.load(MODEL_PATH)
tfidf_vectorizer = joblib.load(TFIDF_PATH)
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)
bert_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

def preprocess_payload(text):
    try:
        text = urllib.parse.unquote(str(text))
        text = html.unescape(text)
        text = " ".join(text.split())
    except:
        pass
    return text

def predict_xss(payload):
    if not payload:
        return "Please enter a payload", 0.0, {}

    start_time = time.time()
    processed = preprocess_payload(payload)
    
    # TF-IDF
    vec_tfidf = tfidf_vectorizer.transform([processed])
    
    # BERT
    inputs = bert_tokenizer([processed], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    vec_bert = csr_matrix(cls_embedding)
    
    # Hybrid
    X_hybrid = hstack([vec_tfidf, vec_bert])
    
    # Predict
    proba = lr_model.predict_proba(X_hybrid)[0]
    is_xss = np.argmax(proba) == 1
    confidence = float(max(proba))
    
    latency = (time.time() - start_time) * 1000
    
    label = "⚠️ XSS DETECTED" if is_xss else "✅ SAFE (BENIGN)"
    color = "red" if is_xss else "green"
    
    details = {
        "Prediction": label,
        "Confidence": f"{confidence:.2%}",
        "Inference Time": f"{latency:.2f} ms",
        "Method": "Hybrid TF-IDF + BERT"
    }
    
    return label, confidence, details

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🛡️ XSS BERT-Hybrid Sentinel
        ### Project Skripsi: Deteksi XSS menggunakan Hybrid BERT Embeddings & TF-IDF
        Masukkan payload (URL, Script, atau Text) di bawah ini untuk mendeteksi potensi serangan XSS.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Payload", 
                placeholder="<script>alert('xss')</script>",
                lines=5
            )
            btn = gr.Button("Analyze Payload", variant="primary")
            
        with gr.Column():
            output_label = gr.Label(label="Result")
            output_json = gr.JSON(label="Detailed Analysis")
            
    gr.Examples(
        examples=[
            ["<script>alert(1)</script>"],
            ["javascript:alert('XSS')"],
            ["<img src=x onerror=alert(1)>"],
            ["Hello world, this is a safe string."],
            ["search?q=query&id=123"]
        ],
        inputs=input_text
    )
    
    btn.click(
        fn=predict_xss, 
        inputs=input_text, 
        outputs=[output_label, gr.Number(visible=False), output_json]
    )

if __name__ == "__main__":
    demo.launch()
