# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app

# Copy the startup script
COPY start.sh ./

# Make the script executable
RUN chmod +x start.sh

# Expose service ports
EXPOSE 8895 8897

# Run the startup script
CMD ["./start.sh"]