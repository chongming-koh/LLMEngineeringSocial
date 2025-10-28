# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 23:18:00 2025

@author: Koh Chong Ming
"""

import os
import gradio as gr
from openai import OpenAI

# --------------------
# Connection to Mebius LLM
# --------------------
os.chdir("C:\\PythonStuff\\keys")
with open("nebius_api_key", "r") as file:
    nebius_api_key = file.read().strip()

nebius_client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=nebius_api_key,
)

model_list = {
    "Meta Llama 3.1 8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Meta Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct",
    "Google Gemma 2 9B": "google/gemma-2-9b-it-fast",
    "Qwen 2.5 72B": "Qwen/Qwen2.5-72B-Instruct",
    "DeepSeek V3 0324": "deepseek-ai/DeepSeek-V3-0324",
    "OpenAI GPT OSS 20B": "openai/gpt-oss-20b",
    "Hermes 4 70B": "NousResearch/Hermes-4-70B",
    "Qwen 3 Coder 480B": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
}

SYSTEM_PROMPT = "You are a helpful assistant"

# --------------------
# takes the chat from Gradio ( chat window history)
# and turns it into the structured format the AI model understands.
# It also adds one special line at the beginning called the system prompt
# --------------------
def to_oai_messages(history_messages, user_message):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history_messages:
        if m.get("role") in ("user", "assistant"):
            msgs.append({"role": m["role"], "content": m.get("content", "")})
    msgs.append({"role": "user", "content": user_message})
    return msgs

# --------------------
# Update from gr.ChatInterface to gr.Chatbot
# Initially I'm using ChatInterface and I cant resolve this error <AttributeError: 'ChatInterface' object has no attribute 'state'>
# Hence, I stop relying on ChatInterface’s hidden state and manage the history yourself with a gr.State
# --------------------
def send(user_message, history_messages, model_name, max_tokens=2056, temperature=0.7):
    print("\n==============================")
    print("send() called")
    print(f"Selected model: {model_name}")
    print(f"Incoming history: {history_messages}")
    print(f"New user message: {user_message}")
    print("==============================\n")

    model_id = model_list.get(model_name, list(model_list.values())[0])
    #print(f"\nModel currently using: {model_id}\n")

    api_messages = to_oai_messages(history_messages, user_message)

    # add user + empty assistant
    history_messages = history_messages + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ""},
    ]
    yield history_messages, history_messages, ""  # (chatbot, state, cleared textbox)

    stream = nebius_client.chat.completions.create(
        model=model_id,
        messages=api_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    partial = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        partial += delta
        history_messages[-1]["content"] = partial
        yield history_messages, history_messages, ""

    yield history_messages, history_messages, ""

# --------------------
# reset function that clears the chat and memory whenever I pick a new model to give a completely fresh start
# --------------------
def on_model_change(new_model):
    print(f"🔄 Model changed to: {new_model} — clearing UI + state")
    return [], []   # clears Chatbot (type='messages') and our State

# --------------------
# User interface
# --------------------
custom_css = """
body { background-color: #b3daff; }
.gradio-container { background-color: #b3daff !important; font-family: 'Segoe UI', Arial, sans-serif; }
"""

with gr.Blocks(css=custom_css, theme="soft") as demo:
    gr.Markdown("## LLM Chat Interface")
    gr.Markdown("Select a model and start chatting below.")

    model_selector = gr.Dropdown(
        label="Select Model",
        choices=list(model_list.keys()),
        value="Meta Llama 3.3 70B",
        interactive=True,
    )

    chatbot = gr.Chatbot(label="Conversation", height=500, type="messages")  # set type='messages'
    state = gr.State([])  # list[dict(role, content)] to match chatbot type

    with gr.Row():
        msg = gr.Textbox(placeholder="Type your message and press Enter", scale=9)
        send_btn = gr.Button("Send", variant="primary", scale=1)

    msg.submit(send, inputs=[msg, state, model_selector], outputs=[chatbot, state, msg])
    send_btn.click(send, inputs=[msg, state, model_selector], outputs=[chatbot, state, msg])

    model_selector.change(on_model_change, inputs=model_selector, outputs=[chatbot, state])

demo.launch(debug=True, inbrowser=True)
