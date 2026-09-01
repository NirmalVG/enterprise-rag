# ingest.py
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load the Document (The Modern, Core Way)
print("Loading document...")
with open("knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()

# We manually wrap our raw text into a LangChain Document object.
# This is exactly what loaders do behind the scenes!
documents = [Document(page_content=text, metadata={"source": "knowledge.txt"})]

# 2. Chunk the Text
print("Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

# 3. Initialize the Embedding Model
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Store in ChromaDB
print("Saving to Vector Database...")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("Ingestion complete! Data is ready for retrieval.")