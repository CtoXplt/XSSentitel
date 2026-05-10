import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from scipy.sparse import csr_matrix, hstack
import joblib
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import urllib.parse
import html
import warnings
warnings.filterwarnings('ignore')

# ====================== 0. PREPROCESSING ======================
def preprocess_payload(text: str) -> str:
    """
    Tahap preprocessing payload XSS:
    1. URL decoding  (contoh: %3Cscript%3E → <script>)
    2. HTML entity decoding (contoh: &lt; → <)
    3. Normalisasi whitespace
    """
    try:
        text = urllib.parse.unquote(str(text))   # URL decode
        text = html.unescape(text)               # HTML entity decode
        text = " ".join(text.split())            # Normalisasi whitespace
    except Exception:
        pass
    return text

# ====================== 1. LOAD DATASET ======================
print("Loading dataset...")
df = pd.read_csv('XSS_dataset.csv')

print("Distribusi label:")
print(df['Label'].value_counts())

total_samples = len(df)
class_distribution = df['Label'].value_counts().to_dict()

X_raw = df['Sentence'].astype(str)
y = df['Label']

# Terapkan preprocessing ke seluruh data
print("Preprocessing payload...")
X = X_raw.apply(preprocess_payload)

# Split data 80:20 — stratified agar proporsi kelas terjaga
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} sampel | Test: {len(X_test)} sampel")

# ====================== 2. TF-IDF FEATURE ======================
print("\nExtracting TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),   # unigram + bigram + trigram
    min_df=2,
    analyzer='char_wb'    # character-level n-gram — lebih robust untuk payload pendek
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf  = tfidf_vectorizer.transform(X_test)
print(f"TF-IDF shape: {X_train_tfidf.shape}")

# ====================== 3. BERT EMBEDDINGS ======================
print("\nExtracting BERT embeddings (ini memerlukan waktu)...")

BERT_MODEL_NAME = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model     = BertModel.from_pretrained(BERT_MODEL_NAME)
bert_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)
print(f"BERT berjalan di: {device}")

def get_bert_embeddings(texts: list, batch_size: int = 32) -> np.ndarray:
    """
    Ekstraksi BERT embeddings menggunakan representasi CLS token.
    CLS token (indeks 0) dari last_hidden_state digunakan sebagai
    representasi kalimat secara keseluruhan.
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT batches"):
        batch = texts[i:i + batch_size]
        inputs = bert_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
    return np.vstack(embeddings)

print("Ekstraksi BERT — Train set...")
X_train_bert = get_bert_embeddings(X_train.tolist())

print("Ekstraksi BERT — Test set...")
X_test_bert = get_bert_embeddings(X_test.tolist())

print(f"BERT embedding shape: {X_train_bert.shape}")

# ====================== 4. HYBRID FEATURE CONCATENATION ======================
# Konversi BERT (dense numpy) → sparse agar kompatibel dengan hstack sparse
X_train_bert_sparse = csr_matrix(X_train_bert)
X_test_bert_sparse  = csr_matrix(X_test_bert)

X_train_hybrid = hstack([X_train_tfidf, X_train_bert_sparse])
X_test_hybrid  = hstack([X_test_tfidf,  X_test_bert_sparse])

print(f"\nHybrid feature shape (train): {X_train_hybrid.shape}")
# Seharusnya: (n_train, 5000 + 768) = (n_train, 5768)

# ====================== 5. TRAINING MODEL ======================
print("\nTraining Logistic Regression on Hybrid Features...")
lr_model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
lr_model.fit(X_train_hybrid, y_train)

# ====================== 6. EVALUASI ======================
y_pred = lr_model.predict(X_test_hybrid)
y_prob = lr_model.predict_proba(X_test_hybrid)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
roc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)
cr_str  = classification_report(y_test, y_pred, target_names=['Benign', 'XSS'])
cr_dict = classification_report(y_test, y_pred, target_names=['Benign', 'XSS'], output_dict=True)

print("\n" + "=" * 60)
print("  HASIL EVALUASI MODEL HYBRID TF-IDF + BERT")
print("=" * 60)
print(f"  Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision  : {prec:.4f}")
print(f"  Recall     : {rec:.4f}")
print(f"  F1-Score   : {f1:.4f}")
print(f"  ROC-AUC    : {roc:.4f}")
print("\nClassification Report:")
print(cr_str)
print("\nConfusion Matrix:")
print(cm)

# ====================== 7. SIMPAN VISUALISASI ======================
output_dir = "evaluation_results"
os.makedirs(output_dir, exist_ok=True)

# --- 7a. Metrics Bar Chart ---
metrics_names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metrics_values = [acc, prec, rec, f1, roc]

plt.figure(figsize=(10, 6))
bars = sns.barplot(x=metrics_names, y=metrics_values, palette='viridis')
plt.ylim(0.95, 1.02)   # zoom ke range relevan agar perbedaan terlihat
plt.title('Evaluation Metrics – Hybrid TF-IDF + BERT', fontsize=14, fontweight='bold')
plt.ylabel('Score')
for i, v in enumerate(metrics_values):
    plt.text(i, v + 0.001, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'metrics_bar_chart.png'), dpi=150)
plt.close()

# --- 7b. Confusion Matrix ---
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', cbar=False,
    xticklabels=['Benign', 'XSS'],
    yticklabels=['Benign', 'XSS']
)
plt.title('Confusion Matrix', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
plt.close()

# --- 7c. Classification Report Heatmap ---
cr_df = pd.DataFrame(cr_dict).T
# Pisahkan baris yang berisi metrik numerik saja (drop accuracy/macro avg/weighted avg jika perlu)
cr_plot = cr_df.loc[['Benign', 'XSS'], ['precision', 'recall', 'f1-score']]

plt.figure(figsize=(8, 4))
sns.heatmap(
    cr_plot.astype(float), annot=True, cmap='YlGn',
    fmt=".4f", vmin=0.95, vmax=1.0,
    linewidths=0.5
)
plt.title('Per-Class Classification Report', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'classification_report.png'), dpi=150)
plt.close()

print(f"\nVisualisasi disimpan di folder: '{output_dir}/'")

# --- 7d. ROC Curve ---
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
# Area bawah kurva
plt.fill_between(fpr, tpr, alpha=0.08, color='#2196F3')
# Kurva utama
plt.plot(fpr, tpr, color='#1565C0', linewidth=2.5,
         label=f'Hybrid TF-IDF + BERT (AUC = {roc:.4f})')
# Garis diagonal baseline (random classifier)
plt.plot([0, 1], [0, 1], color='#BDBDBD', linewidth=1.2,
         linestyle='--', label='Random Classifier (AUC = 0.5000)')
# Titik optimal — threshold dengan jarak terpendek ke (0,1)
optimal_idx   = np.argmax(tpr - fpr)
optimal_fpr   = fpr[optimal_idx]
optimal_tpr   = tpr[optimal_idx]
optimal_thresh = thresholds[optimal_idx]
plt.scatter(optimal_fpr, optimal_tpr, color='#E53935', zorder=5, s=80,
            label=f'Optimal Threshold = {optimal_thresh:.4f}\n'
                  f'TPR = {optimal_tpr:.4f}  |  FPR = {optimal_fpr:.4f}')
plt.annotate(
    f'({optimal_fpr:.4f}, {optimal_tpr:.4f})',
    xy=(optimal_fpr, optimal_tpr),
    xytext=(optimal_fpr + 0.06, optimal_tpr - 0.06),
    fontsize=9, color='#E53935',
    arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.2)
)

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
plt.title('ROC Curve — Hybrid TF-IDF + BERT Embeddings', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')

# Anotasi AUC di sudut kiri atas
plt.text(0.03, 0.92, f'AUC = {roc:.4f}',
         fontsize=13, fontweight='bold', color='#1565C0',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#1565C0', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
plt.close()
print("  ✓ roc_curve.png")

# --- 7e. Distribusi Kelas Dataset ---
labels_dist  = ['XSS', 'Benign']
values_dist  = [int(class_distribution.get(1, 0)), int(class_distribution.get(0, 0))]
colors_dist  = ['#EF5350', '#42A5F5']
explode_dist = (0.04, 0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Pie chart ---
wedges, texts, autotexts = axes[0].pie(
    values_dist, labels=labels_dist, autopct='%1.2f%%',
    colors=colors_dist, explode=explode_dist,
    startangle=90, textprops={'fontsize': 12}
)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight('bold')
axes[0].set_title('Distribusi Kelas Dataset', fontsize=13, fontweight='bold', pad=14)

# --- Bar chart ---
bars_dist = axes[1].bar(labels_dist, values_dist, color=colors_dist,
                         edgecolor='white', linewidth=1.2, width=0.45)
axes[1].set_ylim(0, max(values_dist) * 1.18)
axes[1].set_ylabel('Jumlah Sampel', fontsize=11)
axes[1].set_title('Jumlah Sampel per Kelas', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1].spines[['top', 'right']].set_visible(False)
for bar, val in zip(bars_dist, values_dist):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 60,
                 f'{val:,}', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

plt.suptitle(f'Analisis Distribusi Dataset XSS_dataset.csv  (Total: {total_samples:,} sampel)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ class_distribution.png")

# --- 7f. Perbandingan 3 Konfigurasi Fitur ---
# Train ulang TF-IDF Only dan BERT Only dengan hyperparameter identik
print("\nMelatih model pembanding untuk perbandingan konfigurasi...")

lr_tfidf_only = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
lr_tfidf_only.fit(X_train_tfidf, y_train)
y_pred_tfidf  = lr_tfidf_only.predict(X_test_tfidf)
y_prob_tfidf  = lr_tfidf_only.predict_proba(X_test_tfidf)[:, 1]

lr_bert_only  = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
lr_bert_only.fit(X_train_bert_sparse, y_train)
y_pred_bert   = lr_bert_only.predict(X_test_bert_sparse)
y_prob_bert   = lr_bert_only.predict_proba(X_test_bert_sparse)[:, 1]

configs = ['TF-IDF Only', 'BERT Only', 'Hybrid TF-IDF + BERT']
accs    = [accuracy_score(y_test, y_pred_tfidf),
           accuracy_score(y_test, y_pred_bert),
           acc]
precs   = [precision_score(y_test, y_pred_tfidf),
           precision_score(y_test, y_pred_bert),
           prec]
recs    = [recall_score(y_test, y_pred_tfidf),
           recall_score(y_test, y_pred_bert),
           rec]
f1s     = [f1_score(y_test, y_pred_tfidf),
           f1_score(y_test, y_pred_bert),
           f1]
rocs    = [roc_auc_score(y_test, y_prob_tfidf),
           roc_auc_score(y_test, y_prob_bert),
           roc]

# Print tabel perbandingan ke terminal
print("\n" + "=" * 70)
print("  PERBANDINGAN PERFORMA TIGA KONFIGURASI FITUR")
print("=" * 70)
print(f"{'Konfigurasi':<28} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
print("-" * 70)
for i, cfg in enumerate(configs):
    marker = " ◀ BEST" if i == 2 else ""
    print(f"{cfg:<28} {accs[i]:>9.4f} {precs[i]:>10.4f} {recs[i]:>8.4f} {f1s[i]:>8.4f} {rocs[i]:>9.4f}{marker}")
print("=" * 70)

# Grouped bar chart
metrics_compare  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
values_compare   = [accs, precs, recs, f1s, rocs]
colors_compare   = ['#78909C', '#5C6BC0', '#E53935']
bar_width        = 0.22
x_compare        = np.arange(len(metrics_compare))

fig, ax = plt.subplots(figsize=(13, 6))

for i, (cfg, color) in enumerate(zip(configs, colors_compare)):
    offset = (i - 1) * bar_width
    vals   = [values_compare[j][i] for j in range(len(metrics_compare))]
    bars_c = ax.bar(x_compare + offset, vals, bar_width,
                    label=cfg, color=color, edgecolor='white',
                    linewidth=0.8, zorder=3)
    # Anotasi nilai di atas bar
    for bar_c, val in zip(bars_c, vals):
        ax.text(bar_c.get_x() + bar_c.get_width() / 2,
                bar_c.get_height() + 0.0008,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=7.5, fontweight='bold', rotation=0)

ax.set_ylim(0.96, 1.010)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Perbandingan Performa Tiga Konfigurasi Fitur\n(TF-IDF Only vs BERT Only vs Hybrid TF-IDF + BERT)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x_compare)
ax.set_xticklabels(metrics_compare, fontsize=11)
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
ax.spines[['top', 'right']].set_visible(False)

# Teks catatan dimensi fitur
ax.text(0.01, 0.02,
        'TF-IDF Only: 5.000 dim  |  BERT Only: 768 dim  |  Hybrid: 5.768 dim\n'
        'Classifier: Logistic Regression (C=1.0, solver=lbfgs, max_iter=1000)',
        transform=ax.transAxes, fontsize=8.5, color='#555555',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='#BDBDBD'))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparison_configurations.png'), dpi=150)
plt.close()
print("  ✓ comparison_configurations.png")

# --- 7g. ROC Curve Overlay 3 Konfigurasi ---
fpr_tfidf, tpr_tfidf, _ = roc_curve(y_test, y_prob_tfidf)
fpr_bert,  tpr_bert,  _ = roc_curve(y_test, y_prob_bert)

plt.figure(figsize=(8, 6))
plt.fill_between(fpr, tpr, alpha=0.07, color='#E53935')
plt.plot(fpr_tfidf, tpr_tfidf, color='#78909C', linewidth=1.8, linestyle='--',
         label=f'TF-IDF Only       (AUC = {rocs[0]:.4f})')
plt.plot(fpr_bert,  tpr_bert,  color='#5C6BC0', linewidth=1.8, linestyle='-.',
         label=f'BERT Only         (AUC = {rocs[1]:.4f})')
plt.plot(fpr, tpr,             color='#E53935', linewidth=2.5,
         label=f'Hybrid TF-IDF + BERT (AUC = {rocs[2]:.4f})')
plt.plot([0, 1], [0, 1], color='#BDBDBD', linewidth=1.0, linestyle='--',
         label='Random Classifier (AUC = 0.5000)')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
plt.title('ROC Curve Overlay — Perbandingan 3 Konfigurasi Fitur', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curve_comparison.png'), dpi=150)
plt.close()
print("  ✓ roc_curve_comparison.png")

print(f"\n✅ Total visualisasi tersimpan di '{output_dir}/':")

# ====================== 8. SIMPAN METRICS.JSON & DATASET_INFO.JSON ======================
from datetime import datetime, timezone

metrics_data = {
    "accuracy":     round(acc,  4),
    "precision":    round(prec, 4),
    "recall":       round(rec,  4),
    "f1_score":     round(f1,   4),
    "roc_auc":      round(roc,  4),
    "last_trained": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
}
with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=4)
print("metrics.json disimpan.")

dataset_info = {
    "total_samples": total_samples,
    "classes": {
        "0": "Benign",
        "1": "XSS"
    },
    "class_distribution": {
        "Benign": int(class_distribution.get(0, 0)),
        "XSS":    int(class_distribution.get(1, 0))
    },
    "features": [
        "TF-IDF (5000 features, char_wb n-gram 1-3)",
        "BERT Embeddings (768 features, CLS token, bert-base-uncased)"
    ],
    "source": "XSS_dataset.csv",
    "split": "80% train / 20% test (stratified)",
    "preprocessing": ["URL decoding", "HTML entity decoding", "whitespace normalization"]
}
with open("dataset_info.json", "w", encoding="utf-8") as f:
    json.dump(dataset_info, f, indent=4)
print("dataset_info.json disimpan.")

# ====================== 9. SIMPAN MODEL & VECTORIZER ======================
joblib.dump(lr_model,         'xss_hybrid_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("\n✅ Semua artifact berhasil disimpan:")
print("   - xss_hybrid_model.pkl")
print("   - tfidf_vectorizer.pkl")
print("   - metrics.json")
print("   - dataset_info.json")
print(f"   - evaluation_results/")
print(f"       ├── metrics_bar_chart.png")
print(f"       ├── confusion_matrix.png")
print(f"       ├── classification_report.png")
print(f"       ├── roc_curve.png                ← Gambar 4.4 skripsi")
print(f"       ├── class_distribution.png       ← Gambar 4.1 skripsi")
print(f"       ├── comparison_configurations.png ← Gambar 4.5 skripsi")
print(f"       └── roc_curve_comparison.png     ← ROC overlay 3 konfigurasi")
