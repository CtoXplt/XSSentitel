---
title: XSSentitel API
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# XSS BERT Embeddings Detection (XSSentitel)

## 🚀 Overview
Project Skripsi: Deteksi serangan **Cross-Site Scripting (XSS)** menggunakan pendekatan Hybrid **TF-IDF** dan **BERT Embeddings**. Sistem ini menggunakan **FastAPI** untuk performa tinggi dan **BERT** untuk pemahaman konteks semantik.

## 🛠️ Features
- **FastAPI Backend**: Implementasi asinkronus yang cepat dan optimal.
- **Hybrid Architecture**: Logistic Regression + TF-IDF + BERT CLS token embeddings.
- **BERT Caching**: Optimasi latensi dengan caching embeddings.
- **Interactive UI**: Gradio demo untuk pengujian payload.

## ⚙️ Installation
1. Clone repository:
   ```bash
   git clone https://github.com/CtoXplt/XSSentitel.git
   cd XSSentitel
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run FastAPI API:
   ```bash
   uvicorn api_fastapi:app --host 0.0.0.0 --port 5000
   ```
4. Run Gradio Demo:
   ```bash
   python app.py
   ```

## 📊 Model Evaluation
Bagian ini menunjukkan hasil analisis performa model Hybrid BERT + TF-IDF yang dikembangkan.

### 📈 Performance Visualizations
| Confusion Matrix | ROC Curve |
| :---: | :---: |
| ![Confusion Matrix](evaluation_results/confusion_matrix.png) | ![ROC Curve](evaluation_results/roc_curve.png) |

| Classification Report | Metrics Bar Chart |
| :---: | :---: |
| ![Classification Report](evaluation_results/classification_report.png) | ![Metrics Bar Chart](evaluation_results/metrics_bar_chart.png) |

### 🔍 Comparison & Distribution
| ROC Comparison | Configuration Comparison |
| :---: | :---: |
| ![ROC Curve Comparison](evaluation_results/roc_curve_comparison.png) | ![Comparison Configurations](evaluation_results/comparison_configurations.png) |

> **Note:** Dataset memiliki distribusi kelas yang seimbang seperti yang ditunjukkan pada plot distribusi kelas.
> ![Class Distribution](evaluation_results/class_distribution.png)

## 📝 Dataset
Dataset yang digunakan berisi payload XSS dan benign string yang telah dipreproses.

## 🤝 Contributing
Project ini dikembangkan sebagai bagian dari tugas akhir (Skripsi). Kritik dan saran sangat diterima.

---
© 2026 XSSentinel Team
