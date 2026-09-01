# main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict

# New Hybrid Search Imports
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()
app = FastAPI(title="Enterprise RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Dense Retriever (Semantic Vector Search)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
dense_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

# 2. Sparse Retriever (Exact Keyword Search)
with open("knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()
    
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents([Document(page_content=text)])
sparse_retriever = BM25Retriever.from_documents(chunks)
sparse_retriever.k = 10

# 3. Hybrid Search (Ensemble)
# Merges keyword and vector results, giving each a 50% weight
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5] 
)

# 4. The Reranker (Cross-Encoder Filter)
# The hybrid results are now piped directly into our high-accuracy reranker
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=cross_encoder, top_n=2)

# We use the ensemble_retriever as the base for the compressor
retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)

# --- KEEP EVERYTHING BELOW THIS LINE EXACTLY AS IT WAS ---
# 5. Init LLM
llm = ChatGroq(model="openai/gpt-oss-20b")

# 3. Memory Step: The Question Reformulator
def format_history(history: List[Dict[str, str]]):
    if not history:
        return "No history."
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

rephrase_template = """Given the following chat history and the user's new question, rewrite the new question into a standalone question that contains all the necessary context. Do not answer it, just rewrite it.

Chat History:
{chat_history}

New Question: {question}
Standalone Question:"""
rephrase_chain = ChatPromptTemplate.from_template(rephrase_template) | llm | StrOutputParser()

def get_standalone_question(inputs: dict):
    history = inputs.get("chat_history", [])
    if len(history) > 0:
        return rephrase_chain.invoke({
            "chat_history": format_history(history),
            "question": inputs["question"]
        })
    return inputs["question"]

# 4. Final RAG Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_template = """Answer the question based ONLY on the following context. If you don't know, say you don't know.

Context:
{context}

Question: {standalone_question}
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)

# We map inputs to the standalone question, then pipe that into the retriever and final prompt
rag_chain = (
    {
        "standalone_question": get_standalone_question,
        "context": get_standalone_question | retriever | format_docs, 
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)

# 5. Schema (Now accepts history!)
class QueryRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, str]] = []

@app.post("/ask")
async def ask_question(request: QueryRequest):
    # We create an async generator function that yields text chunks as they arrive
    async def generate():
        # .astream() automatically streams the output of the StrOutputParser
        async for chunk in rag_chain.astream({
            "question": request.question,
            "chat_history": request.chat_history
        }):
            yield chunk

    # Return the stream with a standard text media type
    return StreamingResponse(generate(), media_type="text/plain")