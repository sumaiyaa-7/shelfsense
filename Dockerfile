FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m shelfsense.seed_db
EXPOSE 8000
CMD ["uvicorn", "shelfsense.api:app", "--host", "0.0.0.0", "--port", "8000"]
