FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app
COPY . .

# Set env variable untuk Flask
ENV FLASK_APP=app.py

# Start command (sama seperti di Railway)
CMD ["gunicorn", "app:application", "--bind", "0.0.0.0:5000"]
