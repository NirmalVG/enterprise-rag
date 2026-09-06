# graph_builder.py
import os
from dotenv import load_dotenv
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_core.documents import Document

load_dotenv()

# 1. Initialize the Extractor
# We use a larger model because structured extraction is complex
llm = ChatGroq(model="openai/gpt-oss-120b")
llm_transformer = LLMGraphTransformer(llm=llm)

text = """
TechCorp was founded by Sarah Jenkins in 2015. 
In 2022, MegaCloud acquired TechCorp for $2 Billion. 
MegaCloud's current CEO is David Chen.
"""

print("1. Reading text and extracting Graph Nodes & Edges...")
docs = [Document(page_content=text)]

# The LLM converts the raw text into structured graph data
graph_documents = llm_transformer.convert_to_graph_documents(docs)

print("\n--- Extracted Entities (Nodes) ---")
for node in graph_documents[0].nodes:
    print(f"[{node.type}] {node.id}")

print("\n--- Extracted Relationships (Edges) ---")
for edge in graph_documents[0].relationships:
    print(f"({edge.source.id}) --[{edge.type}]--> ({edge.target.id})")