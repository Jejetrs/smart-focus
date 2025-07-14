FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .

# Pastikan variabel berikut sesuai dengan instance Flask kamu
# Jika di app.py kamu ada `app = Flask(__name__)`, gunakan app:app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
