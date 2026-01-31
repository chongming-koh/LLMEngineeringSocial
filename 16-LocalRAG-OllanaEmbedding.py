# -*- coding: utf-8 -*-
"""
Created on Sat 31 Jan 2026

@author: Koh Chong Ming

Local RAG Pipeline (LlamaCPP + Ollama + Chroma)

"""
import os
import glob
from tqdm import tqdm 
# imports for langchain and Chroma and plotly

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings


#Chroma storage
db_name = "chromaOllama_Treasurydb"

# ==============================================
# 1. Document Loading (PDF + TXT) with Progress Bar
# ==============================================

knowledge_base_path = "knowledge-base"
folders = glob.glob(os.path.join(knowledge_base_path, "*"))

documents = []

print("Loading documents...")
for folder in tqdm(folders, desc="Loading folders", ncols=100):
    doc_type = os.path.basename(folder)

    # Load PDFs
    pdf_loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()

    # Load Text Files
    text_loader = DirectoryLoader(folder, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    text_docs = text_loader.load()

    for doc in pdf_docs + text_docs:
        doc.metadata["doc_type"] = doc_type

    documents.extend(pdf_docs + text_docs)

print(f"Loaded {len(documents)} total documents from {knowledge_base_path}\n")

# ==============================================
# 2. Chunking
# ==============================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Check if correct set of files are loaded correctly
doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"Document types found: {', '.join(doc_types)}\n")

# ==============================================
# 3. Vectorization with Ollama Embedding Model (Local)
# ==============================================

embedding_model = "embeddinggemma:latest"  # Ollama embedding model


if os.path.exists(db_name):
    print("Existing Chroma datastore found — deleting old collection...")
    Chroma(persist_directory=db_name, embedding_function=None).delete_collection()
else:
    os.makedirs(db_name, exist_ok=True)

print(f"Generating embeddings locally with Ollama model '{embedding_model}'...")
embeddings = OllamaEmbeddings(model=embedding_model)

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vector storage created and saved to '{db_name}' with {vectorstore._collection.count()} vectors")

# ==============================================
# 4) Check the data length (Optional, comment the block to skip)
# ==============================================


lengths = [len(d.page_content) for d in chunks]
print("Chunks:", len(chunks))
print("Max chars:", max(lengths))
print("Top 5 chars:", sorted(lengths, reverse=True)[:5])

