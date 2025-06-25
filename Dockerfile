# Use the official Python 3.10 slim image as the base image
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /mcp

# Install Poetry
RUN pip install poetry

# Copy the application code
COPY . /mcp

# Install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root

RUN pip install gunicorn

# Run the application
# CMD ["gunicorn", "-w", "4", "--bind", "0.0.0.0:$FLASK_PORT", "run:app"]
CMD ["make", "start"]
