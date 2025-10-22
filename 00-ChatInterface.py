# -*- coding: utf-8 -*-
"""
Created on Sat Oct 04 23:18:00 2025

@author: Koh Chong Ming
"""

import os
import gradio as gr
from openai import OpenAI

#Find the key file

os.chdir("C:\\PythonStuff\\keys")
cwd = os.getcwd() 

with open("nebius_api_key", "r") as file:
    nebius_api_key = file.read().strip()

os.environ["NEBIUS_API_KEY"] = nebius_api_key

# Nebius uses the same OpenAI() class, but with additional details
nebius_client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

llama_8b_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
llama_70b_model ="meta-llama/Llama-3.3-70B-Instruct"
gemma_9b_model = "google/gemma-2-9b-it-fast"
Qwen2_5_72B_model = "Qwen/Qwen2.5-72B-Instruct"
DeepSeek_V33024 ="deepseek-ai/DeepSeek-V3-0324"
openai_20b = "openai/gpt-oss-20b"
Hermes_4_70B_model ="NousResearch/Hermes-4-70B"
Qwen_Qwen3_Coder="Qwen/Qwen3-Coder-480B-A35B-Instruct"

system_prompt = "You are a helpful assistant"

# ==============================================
# 1. Chat function
# To stream response, set use_stream as True
# ==============================================

def chat(message, history, 
         client=nebius_client,
         max_tokens=2056,
         temperature=0.7,
         model=gemma_9b_model,
         use_stream=True):
    
    #system_message = "You are a helpful assistant that answers questions using context if provided."

    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]

    # --------------------------
    # STREAMING MODE
    # --------------------------
    if use_stream:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )

        response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            response += delta
            yield response

    # --------------------------
    # NON-STREAMING MODE
    # --------------------------
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        response = completion.choices[0].message.content
        yield response  
        
# ==============================================
# 2. Use Gradio to invoke the chat function
# ==============================================
custom_css = """
body {
    background-color: #b3daff; /* baby blue */
}

.gradio-container {
    background-color: #b3daff !important;
    font-family: 'Segoe UI', Arial, sans-serif;
}

.message {
    border-radius: 16px !important;
    padding: 10px 14px !important;
    font-size: 16px;
    line-height: 1.5;
}

.user {
    background-color: #e6f3ff !important;
    color: #003366 !important;
}

.assistant {
    background-color: #f0f8ff !important;
    color: #002244 !important;
    border: 1px solid #cce0ff;
}

footer, .footer {
    background-color: #b3daff !important;
}

button, .gr-button {
    background-color: #80c1ff !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    transition: 0.3s;
}

button:hover, .gr-button:hover {
    background-color: #66b3ff !important;
}
"""

gr.ChatInterface(
    fn=chat,
    type="messages",
    title="🛫 LLM Chat Interface",
    description="Ask about code or a simple conversation",
    theme="soft",
    css=custom_css,
).launch(debug=True, inbrowser=True)