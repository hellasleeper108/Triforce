FROM python:3.11-slim

WORKDIR /app

COPY python-odin/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY python-odin/main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
