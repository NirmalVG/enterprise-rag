FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your scripts and knowledge files (knowledge.txt, sample.pdf)
COPY . .

# Generate the ChromaDB and chunks.pkl file directly inside the cloud container
RUN python ingest.py

EXPOSE 8000

# Start the server using Gunicorn for production stability
CMD ["gunicorn", "main:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]