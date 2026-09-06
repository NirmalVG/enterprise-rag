# ingest.py
import pickle
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("1. Loading Documents...")
# Load a PDF
pdf_loader = PyPDFLoader("sample.pdf")
pdf_docs = pdf_loader.load()

# Load a Web Page
web_loader = WebBaseLoader("https://react.dev/learn")
web_docs = web_loader.load()

# Combine all loaded documents
all_docs = pdf_docs + web_docs

print(f"2. Chunking {len(all_docs)} pages...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(all_docs)

print("3. Saving Vectors to ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# We clear the old database to avoid duplicates during testing
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("4. Exporting raw chunks for BM25...")
# Save the chunks to a highly compressed binary file for lightning-fast server startup
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Ingestion complete! Data is ready for retrieval.")