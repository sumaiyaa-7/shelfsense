FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m shelfsense.seed_db
EXPOSE 8000
# Hosting platforms like Render pass the port in $PORT; default to 8000 locally
CMD ["sh", "-c", "uvicorn shelfsense.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
