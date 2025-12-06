FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire triforce package
COPY triforce /app/triforce
# Add . to PYTHONPATH so we can import triforce
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["python", "triforce/odin/main.py"]
