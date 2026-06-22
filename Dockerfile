FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run sets the PORT environment variable. We read it or default to 8080.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
