FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/backups /app/checkpoints /app/logs /app/models
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "main.py"]
