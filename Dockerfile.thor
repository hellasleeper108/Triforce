
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY triforce/ /app/triforce/

# Set env
# Set env
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "triforce/thor/main.py"]

