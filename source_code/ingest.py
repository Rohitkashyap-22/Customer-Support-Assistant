import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the API Key
load_dotenv()

def ingest_document():
    print("Step 1: Loading PDF...")
    loader = PyPDFLoader("sample.pdf")
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    print("\nStep 2: Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")

    print("\nStep 3: Generating embeddings using Gemini...")
    # UPDATED to the correct, currently active Google embedding model!
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    print("\nStep 4: Saving to ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("\nSUCCESS! Ingestion complete. Your data is now an AI-searchable database.")

if __name__ == "__main__":
    ingest_document()