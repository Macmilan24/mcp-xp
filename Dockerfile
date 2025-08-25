# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app

# Expose service port
EXPOSE 8000

# Default envs (Compose overrides as needed)
ENV PORT=8000

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]