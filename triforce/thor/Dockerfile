FROM python:3.10-slim

WORKDIR /app

# Install system deps for metrics/sandbox if needed
RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire triforce package
COPY triforce /app/triforce
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "triforce/thor/main.py"]
