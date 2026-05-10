# Gunakan image Python yang stabil
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Set working directory
WORKDIR /app

# Install dependencies sistem (kalau butuh build library tertentu)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy semua file project
COPY . .

# Expose port yang digunakan Hugging Face Spaces
EXPOSE 7860

# Jalankan aplikasi menggunakan Uvicorn
# Kita pake port 7860 karena itu default-nya HF Spaces
CMD ["uvicorn", "api_fastapi:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
