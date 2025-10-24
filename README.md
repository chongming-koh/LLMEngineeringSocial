
![Summit](https://github.com/user-attachments/assets/cfde5967-1d90-476a-995d-066277f219d4)

 # Welcome to My LLM Engineering Portfolio

Hi, I'm Koh Chong Ming, a transaction banking product expert and LLM engineering enthusiast exploring how AI can power the next generation of intelligent financial systems.

This repository showcases my journey in large language model (LLM) engineering, including my personal projects. My focus is on applying these technologies to solve real-world challenges in payments automation, cash management and corporate treasury.

**My Approach**

Every project here is guided by one question: “How can LLMs make human work smarter, faster, and more insightful?” You’ll find experiments that blend **AI reasoning, retrieval-augmented generation (RAG), agentic workflows, and synthetic data generation** which I want to master and apply to payments, corporate treasury and transaction banking.

🚀 **Featured Themes**

- **LLM Workflows & Agents:** experimenting with LangChain, and custom tool-calling frameworks

- **Vectorized RAG Pipelines:** building scalable retrieval systems using embeddings, Chroma/FAISS, vectorizing and retireval with cloud host LLM or local LLM (like Llama CPP and Ollama) for privacy

- **Synthetic Data Generation:** creating realistic datasets for payments, treasury, and automation testing

## 00-ChatInterface.py

LLM Chat Interface is a lightweight, customizable chat UI built with Gradio and powered by Nebius AI’s OpenAI-compatible API.
It enables seamless interaction with large language models like Llama 3.3 70B, Gemma 2 9B, and DeepSeek V3, supporting both streaming and non-streaming responses.

**Features**

- Multi-model flexibility: Easily switch between Llama, Gemma, Qwen, DeepSeek, and more.
- Real-time streaming replies: Token-by-token updates for a natural chat flow.
- Customizable UI: Soft blue theme with rounded message bubbles and interactive buttons.
- Easy integration: Built entirely in Python with gradio and openai SDK — no complex setup.
- Secure API handling: Automatically loads API keys from a local environment file.

**How It Works**

- Launches a browser-based chat interface titled “LLM Chat Interface”.
- Sends user messages through Nebius’s OpenAI endpoint.
- Streams responses dynamically using the selected model and parameters (temperature, max_tokens, etc.).

This project is experimenting with LLM chatbots, custom AI assistants, or embedded conversational UIs that run locally or on the web.


## 09-ToolCall-ConcertTicketAssistant.py

AITICKETS is an interactive AI-powered ticketing portal that simulates a smart concert-ticket customer service assistant.
Built with Gradio and the Nebius OpenAI-compatible API, it demonstrates manual tool-calling logic and no LangChain required to perform reasoning, function execution, and multi-turn dialogue.

**Key Capabilities**

- Smart Ticket Queries: Instantly fetch concert ticket prices for global destinations.

- Automated Booking: Generate live booking confirmations and receipts.

- Currency Conversion: Convert USD/EUR prices to SGD using predefined exchange rates.

- Reasoning & Tool-Calling: Handles tool invocation manually via JSON parsing for clear visibility into the LLM’s decision flow.

- Custom Chat UI: Baby-blue themed interface with clean rounded chat bubbles, styled for a modern conversational experience.

**Tech Highlights**

- Uses OpenAI.chat.completions.create() with built-in tool definitions and calls.

- Demonstrates structured reasoning loops starting with initial LLM reply → tool call → response integration → final assistant reply.

- Fully self-contained Python script with no external frameworks beyond gradio and openai.

**Perfect For Exploring**

- How to build tool-using LLM agents without LangChain.

- Designing custom AI customer service systems.

- Building branded chat interfaces for AI-powered commerce platforms.
