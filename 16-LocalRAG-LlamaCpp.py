# -*- coding: utf-8 -*-
"""
Created on Sat 31 Jan 2026

@author: Koh Chong Ming

Local RAG Pipeline (LlamaCPP + Ollama + Chroma)

"""

import gradio as gr

# imports for langchain and Chroma

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

#Chroma storage
db_name = "chromaOllama_Treasurydb"


# ==============================================
# Load existing Vector data created by embedded LLM previously from Chroma store
# Data persists in Chroma
# When i do not want to go through the vectorization again.
# ==============================================

embedding_model = "embeddinggemma:latest"  # Ollama embedding model
embeddings = OllamaEmbeddings(model=embedding_model)
vectorstore = Chroma(
    persist_directory=db_name,
    embedding_function=embeddings
)

print("✅ Loaded existing vectorstore!")
print(f"Stored documents: {vectorstore._collection.count()}")


# ==============================================
# 4. Retrieval and Local Chat with LlamaCPP
# ==============================================

llama_chat_model_path = r"C:\\LlamaCPP\\models\\Llama-3.2-1B-Instruct-Q8_0.gguf"

llm = LlamaCpp(
    model_path=llama_chat_model_path,
    n_ctx=8192,
    n_batch=256,
    temperature=0.7,
    verbose=False
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# ==============================================
# 4b. Alternative Retrieval and Local Chat with Ollama Model Llama3.2
# Uncomment to use Ollama
# ==============================================
'''
from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0.7,
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)
'''
# ==============================================
# 5. Gradio Chat UI
# ==============================================

def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

view = gr.ChatInterface(chat, type="messages", title="Local RAG Assistant (Ollama Embedding + LlamaCPP/Ollama)").launch(inbrowser=True)