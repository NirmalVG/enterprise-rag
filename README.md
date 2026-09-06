# Enterprise RAG Project

A full-stack Retrieval-Augmented Generation (RAG) application with a FastAPI backend and a Next.js frontend. The backend combines semantic vector search, BM25 keyword search, cross-encoder reranking, conversational question reformulation, and streamed LLM responses.

## Project Structure

```text
rag-project/
├── rag-backend/     # FastAPI API, ingestion scripts, Chroma database
└── rag-frontend/    # Next.js chat interface
```

## Prerequisites

- Python 3.11+
- Node.js 20+
- A Groq API key

## Configuration

Create `rag-backend/.env` with:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env` or API keys.

## Run the Backend

```bash
cd rag-backend
python -m venv venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

Install the backend dependencies listed in `rag-backend/requirements.txt`, then start the API:

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API is available at `http://localhost:8000`. The chat endpoint is `POST /ask`.

### Rebuild the Knowledge Base

Run ingestion from `rag-backend/` when the source documents change:

```bash
python ingest.py
```

This rebuilds `chroma_db/` and generates `chunks.pkl`. The ingestion script currently loads `sample.pdf` and content from `https://react.dev/learn`.

## Run the Frontend

In a second terminal:

```bash
cd rag-frontend
npm install
npm run dev
```

Open `http://localhost:3000` in a browser. The frontend uses `http://localhost:8000` by default. To use another backend URL, create `rag-frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Docker

Build and run the backend from the backend directory:

```bash
cd rag-backend
docker build -t enterprise-rag-backend .
docker run --env-file .env -p 8000:8000 enterprise-rag-backend
```

The frontend can still be run locally with `npm run dev`.

## API Example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is retrieval-augmented generation?","chat_history":[]}'
```
