# Dockerfile for the standalone MCP Gateway Service

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p app/bioblend_server

COPY app/__init__.py app/
COPY app/bioblend_server/__init__.py app/bioblend_server/

COPY app/bioblend_server/server.py app/bioblend_server/
COPY app/bioblend_server/__main__.py app/bioblend_server/
COPY app/bioblend_server/utils.py app/bioblend_server/
COPY app/bioblend_server/galaxy.py app/bioblend_server/

COPY app/log_setup.py app/

EXPOSE 8001

CMD ["python", "-m", "app.bioblend_server.__main__"]