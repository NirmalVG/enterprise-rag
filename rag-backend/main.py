# main.py
import os
import pickle
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict

# Hybrid Search Imports
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# Vector & Embedding Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Prompt & Chain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Reranker Imports — now hosted via Cohere instead of a local torch model
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Load environment variables
load_dotenv()

app = FastAPI(title="Enterprise RAG API: Multi-Source Edition")

app.add_middleware(
    CORSMiddleware,
    # Replace the second URL with your actual live domain later
    allow_origins=["http://localhost:3000", "https://your-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Dense Retriever (Semantic Vector Search)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
dense_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

# 2. Sparse Retriever (Exact Keyword Search)
# Instantly load the pre-processed chunks we generated in ingest.py
with open("chunks.pkl", "rb") as file:
    chunks = pickle.load(file)

sparse_retriever = BM25Retriever.from_documents(chunks)
sparse_retriever.k = 10

# 3. Hybrid Search (Ensemble)
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)

# 4. The Reranker — Cohere's hosted Rerank API
# No local model load = no second torch model resident in memory.
# Requires COHERE_API_KEY set in your environment (Render dashboard -> Environment).
compressor = CohereRerank(model="rerank-v3.5", top_n=2)
retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)

# 5. Initialize LLM
llm = ChatGroq(model="openai/gpt-oss-20b")

# 6. Memory Step: The Question Reformulator
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

# 7. Final Generation Chain setup (Decoupled from retriever for citations)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_template = """Answer the question based ONLY on the following context. If you don't know, say you don't know.

Context:
{context}

Question: {standalone_question}
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)
qa_chain = qa_prompt | llm | StrOutputParser()

# 8. Agentic Routing Chains
router_template = """Determine if the user's message requires looking up factual information, or if it is just a casual greeting/conversation. 
Respond with exactly one word: 'SEARCH' or 'CASUAL'.

Message: {question}
Decision:"""
router_chain = ChatPromptTemplate.from_template(router_template) | llm | StrOutputParser()

casual_template = """You are a helpful, friendly AI assistant. Respond conversationally to the user.
User: {question}
"""
casual_chain = ChatPromptTemplate.from_template(casual_template) | llm | StrOutputParser()


# 9. Define the Request Schema
class QueryRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, str]] = []

# 10. Expose the API Endpoint with Routing, Streaming, and Citations
@app.post("/ask")
async def ask_question(request: QueryRequest):
    async def generate():
        # Step A: Classify intent
        decision = router_chain.invoke({"question": request.question}).strip().upper()

        # Step B: Route dynamically
        if "CASUAL" in decision:
            # Skip the database, stream friendly response
            async for chunk in casual_chain.astream({"question": request.question}):
                yield chunk
        else:
            # Step C: RAG Pipeline - Resolve conversational memory
            standalone_q = get_standalone_question({
                "question": request.question,
                "chat_history": request.chat_history
            })

            # Step D: Retrieve documents and extract unique source metadata
            docs = retriever.invoke(standalone_q)
            sources = set([doc.metadata.get("source", "Unknown document") for doc in docs])

            # Step E: Stream the factual LLM response
            async for chunk in qa_chain.astream({
                "context": format_docs(docs),
                "standalone_question": standalone_q
            }):
                yield chunk

            # Step F: Append the citations to the stream
            if sources:
                yield "\n\n**Sources:**\n"
                for source in sources:
                    yield f"- {source}\n"

    return StreamingResponse(generate(), media_type="text/plain")