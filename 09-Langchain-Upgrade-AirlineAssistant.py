# -*- coding: utf-8 -*-
"""
Created on Oct 23 19:16:22 2025

@author: Koh Chong Ming

Modify existing Customer Service UI tool call with Langchain framework
"""

# flightai_langchain_app.py
# One-file app: LangChain tools + Nebius (OpenAI-compatible) + Gradio UI

# flightai_langchain_app_simple.py
# One-file app: LangChain tools + Nebius (OpenAI-compatible) + Gradio UI (simplified)

import os
import json
import gradio as gr

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool

# =========================
# 0) Keys + Nebius wiring
# =========================

# Retrieve the keys for connection
os.chdir("C:\\PythonStuff\\keys")
with open("nebius_api_key", "r") as file:
    nebius_api_key = file.read().strip()
os.environ["NEBIUS_API_KEY"] = nebius_api_key

# Model names (same as your originals)
llama_8b_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
llama_70b_model = "meta-llama/Llama-3.3-70B-Instruct"
gemma_9b_model = "google/gemma-2-9b-it-fast"
Qwen2_5_72B_model = "Qwen/Qwen2.5-72B-Instruct"
DeepSeek_V33024 = "deepseek-ai/DeepSeek-V3-0324"
openai_20b = "openai/gpt-oss-20b"
Hermes_4_70B_model = "NousResearch/Hermes-4-70B"

# Direct, simple LLM instance (no factory function)
llm = ChatOpenAI(
    model=llama_70b_model,                 # switch here if you want another model
    temperature=0.7,
    api_key=os.environ["NEBIUS_API_KEY"],
    base_url="https://api.studio.nebius.ai/v1/",
)

# =========================
# 1) System prompt
# =========================

system_prompt = (
    "You are a helpful assistant for an airline called FlightAI. "
    "Answer general questions normally. "
    "ONLY call get_ticket_price if the user explicitly asks about ticket prices or destinations. "
    "Call book_tickets if the user asks to book or purchase tickets. "
    "Call convert_currency if the user asks to convert prices or currencies. "
    "Do NOT call any tools for unrelated questions like greetings, names, or general chat. "
    "Keep answers short and courteous."
)

# =========================
# 2) Domain data & functions (unchanged)
# =========================

# Ticket price data in SGD
ticket_prices = {"london": 1040, "paris": 1170, "tokyo": 1820, "berlin": 650}

# Currency conversion rates
conversion_rates = {"USD": 1.30, "EUR": 1.50}

def format_currency(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return str(value)
    return f"{value:,.2f}"

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    price = ticket_prices.get(city)
    if price:
        return {"destination_city": city, "price_sgd": price}
    else:
        return {"error": "Unknown destination."}

def book_tickets(destination_city, number_of_tickets):
    print(f"Tool book_tickets called for {destination_city} ({number_of_tickets} tickets)")
    city = destination_city.lower()
    if city not in ticket_prices:
        return {"error": "Destination not available."}
    total_price = ticket_prices[city] * number_of_tickets
    receipt = (
        f"\nFlightAI Booking Receipt\n"
        f"---------------------------------\n"
        f"Destination: {city.title()}\n"
        f"Tickets Purchased: {number_of_tickets}\n"
        f"Price per Ticket: ${format_currency(ticket_prices[city])} SGD\n"
        f"Total Amount: ${format_currency(total_price)} SGD\n"
        f"---------------------------------\n"
        f"Thank you for choosing FlightAI. Have a pleasant journey!"
    )
    return {
        "destination_city": city,
        "tickets": number_of_tickets,
        "total_price_sgd": total_price,
        "receipt": receipt
    }

def convert_currency(amount, from_currency):
    print(f"Tool convert_currency called: {amount} {from_currency} -> SGD")
    try:
        amount = float(amount)  # ensure numeric conversion
    except ValueError:
        return {"error": "Amount must be a number."}

    rate = conversion_rates.get(from_currency.upper())
    if rate:
        converted = amount * rate
        receipt = (
            f"\nCurrency Conversion Receipt\n"
            f"---------------------------------\n"
            f"Amount: {format_currency(amount)} {from_currency.upper()}\n"
            f"Conversion Rate: 1 {from_currency.upper()} = {rate} SGD\n"
            f"Converted Amount: {format_currency(converted)} SGD\n"
            f"---------------------------------\n"
            f"Thank you for using FlightAI Currency Converter."
        )
        return {
            "amount_sgd": round(converted, 2),
            "rate": rate,
            "receipt": receipt
        }
    else:
        return {"error": "Unsupported currency."}

# =========================
# 3) LangChain tool wrappers
# =========================

@tool("get_ticket_price")
def lc_get_ticket_price(destination_city: str) -> dict:
    """Get the price of a return ticket to a destination city."""
    return get_ticket_price(destination_city)

@tool("book_tickets")
def lc_book_tickets(destination_city: str, number_of_tickets: int) -> dict:
    """Book tickets for a destination and number of passengers."""
    return book_tickets(destination_city, number_of_tickets)

@tool("convert_currency")
def lc_convert_currency(amount: float, from_currency: str) -> dict:
    """Convert USD or EUR into SGD using predefined rates."""
    return convert_currency(amount, from_currency)

TOOLS = [lc_get_ticket_price, lc_book_tickets, lc_convert_currency]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# Bind tools once (keeps things minimal)
llm_with_tools = llm.bind_tools(TOOLS)

# =========================
# 4) Tool-call handler (LangChain-native)
# =========================

def handle_tool_calls_lc(ai_msg: AIMessage):
    """Execute all tool calls emitted by the model and return a list of ToolMessage(s)."""
    tool_messages = []
    if not getattr(ai_msg, "tool_calls", None):
        return tool_messages

    for call in ai_msg.tool_calls:
        name = call.get("name")
        args = call.get("args", {})
        call_id = call.get("id")
        tool = TOOLS_BY_NAME.get(name)

        if tool is None:
            result = {"error": f"Unknown tool: {name}"}
        else:
            result = tool.invoke(args)  # passes **args into your wrapped function

        tool_messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call_id))

    return tool_messages

# =========================
# 5) Chat function for Gradio
# =========================

def chat(
    message,
    history,
    system_prompt_text=system_prompt,
    max_model_steps=4,   # safety cap to avoid infinite loops
):
    # Build conversation
    msgs = [SystemMessage(system_prompt_text)]

    for turn in (history or []):
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content))
        elif role in ("assistant", "ai"):
            msgs.append(AIMessage(content))

    msgs.append(HumanMessage(message))

    # First call
    ai = llm_with_tools.invoke(msgs)

    # Tool loop
    steps = 0
    while getattr(ai, "tool_calls", None) and steps < max_model_steps:
        print("Model requested tool(s):", ai.tool_calls)
        tool_messages = handle_tool_calls_lc(ai)
        msgs.extend([ai] + tool_messages)
        ai = llm_with_tools.invoke(msgs)
        steps += 1

    return ai.content

# =========================
# 6) Gradio UI
# =========================

if __name__ == "__main__":
    gr.ChatInterface(
        fn=chat,
        type="messages",
        title="FlightAI Customer Support",
        description="Ask about flights, ticket prices, or make bookings in a friendly baby-blue chat window",
        theme="soft",
    ).launch(debug=True, inbrowser=True)
