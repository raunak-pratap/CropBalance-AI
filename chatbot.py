"""
chatbot.py
----------
CropBalanceAI chatbot powered by Claude.
Run: streamlit run chatbot.py
"""

import streamlit as st
import anthropic
import requests
import json
import base64
from datetime import datetime

# ── Config ─────────────────────────────────────
API_BASE   = "http://localhost:8000"   # Your running FastAPI
CLAUDE_MODEL = "claude-sonnet-4-5"

client = anthropic.Anthropic()         # Reads ANTHROPIC_API_KEY from .env

# ── Tools Claude can call ──────────────────────
TOOLS = [
    {
        "name": "predict_crop_price",
        "description": "Predict crop prices for next 30 days. Use when farmer asks about future prices.",
        "input_schema": {
            "type": "object",
            "properties": {
                "crop":  {"type": "string", "description": "Crop name e.g. wheat, rice, tomato"},
                "state": {"type": "string", "description": "Indian state e.g. Punjab, Maharashtra"},
            },
            "required": ["crop", "state"],
        },
    },
    {
        "name": "get_live_price",
        "description": "Get today's mandi price for a crop. Use when farmer asks current price.",
        "input_schema": {
            "type": "object",
            "properties": {
                "crop":  {"type": "string"},
                "state": {"type": "string"},
            },
            "required": ["crop", "state"],
        },
    },
    {
        "name": "detect_disease",
        "description": "Detect disease from a crop leaf image. Use when farmer uploads a photo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_base64": {"type": "string", "description": "Base64 encoded image"},
            },
            "required": ["image_base64"],
        },
    },
    {
        "name": "get_price_history",
        "description": "Get historical prices for charting. Use when farmer asks about past prices.",
        "input_schema": {
            "type": "object",
            "properties": {
                "crop":  {"type": "string"},
                "state": {"type": "string"},
                "days":  {"type": "integer", "description": "Number of days of history"},
            },
            "required": ["crop", "state"],
        },
    },
]

SYSTEM_PROMPT = """You are CropBalanceAI, a helpful farming assistant for Indian farmers.
You help farmers with:
- Crop price predictions (next 30 days)
- Current mandi prices
- Crop disease detection from photos
- Historical price trends

Rules:
- Always respond in the same language the farmer uses (Hindi or English)
- When predicting prices, always mention the unit (₹ per quintal)
- For diseases, always give severity and treatment in simple language
- Be warm, friendly, and use simple words farmers understand
- If a farmer uploads a photo, always use the detect_disease tool
- Always mention which state you're using for the prediction

You have access to real AI models trained on Indian agricultural data.
"""


# ── Tool execution ──────────────────────────────
def execute_tool(tool_name: str, tool_input: dict) -> str:
    try:
        if tool_name == "predict_crop_price":
            resp = requests.post(f"{API_BASE}/predict", json={
                "crop":         tool_input["crop"],
                "state":        tool_input["state"],
                "days_history": 90,
            })
            data = resp.json()
            forecast = data.get("forecast", [])
            # Summarise first 7 days
            summary = "\n".join([
                f"  {f['date']}: ₹{f['price_inr']:.0f}/quintal"
                for f in forecast[:7]
            ])
            avg = sum(f["price_inr"] for f in forecast) / len(forecast)
            return f"30-day forecast for {tool_input['crop']} in {tool_input['state']}:\n{summary}\n...\nAverage predicted price: ₹{avg:.0f}/quintal"

        elif tool_name == "get_live_price":
            resp = requests.get(f"{API_BASE}/prices/live", params={
                "crop":  tool_input["crop"],
                "state": tool_input["state"],
            })
            data = resp.json()
            return (
                f"Today's price for {data['crop']} in {data['state']}:\n"
                f"  Modal price: ₹{data['modal_price_inr']}/quintal\n"
                f"  Min: ₹{data['min_price_inr']} | Max: ₹{data['max_price_inr']}\n"
                f"  Market arrivals: {data['arrivals_tonnes']} tonnes"
            )

        elif tool_name == "detect_disease":
            image_bytes = base64.b64decode(tool_input["image_base64"])
            resp = requests.post(
                f"{API_BASE}/disease/detect",
                files={"image": ("leaf.jpg", image_bytes, "image/jpeg")},
            )
            data = resp.json()
            return (
                f"Disease detected: {data['disease'].replace('_', ' ').title()}\n"
                f"Confidence: {data['confidence']*100:.1f}%\n"
                f"Severity: {data['severity'].upper()}\n"
                f"Treatment (EN): {data['treatment_en']}\n"
                f"Treatment (HI): {data['treatment_hi']}"
            )

        elif tool_name == "get_price_history":
            resp = requests.get(
                f"{API_BASE}/prices/history/{tool_input['crop']}",
                params={"state": tool_input["state"], "days": tool_input.get("days", 30)},
            )
            data = resp.json()
            records = data.get("records", [])
            if records:
                prices = [r["modal_price"] for r in records]
                return (
                    f"Price history for {tool_input['crop']} in {tool_input['state']} "
                    f"({tool_input.get('days', 30)} days):\n"
                    f"  Highest: ₹{max(prices):.0f}/quintal\n"
                    f"  Lowest:  ₹{min(prices):.0f}/quintal\n"
                    f"  Average: ₹{sum(prices)/len(prices):.0f}/quintal"
                )

    except Exception as e:
        return f"Error calling {tool_name}: {str(e)}"

    return "Tool not found"


# ── Claude with tools ──────────────────────────
def chat_with_claude(messages: list, image_b64: str = None) -> str:
    # If image uploaded, add it to the last message
    if image_b64:
        messages[-1]["content"] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
            {"type": "text",  "text": messages[-1]["content"]},
        ]

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
    )

    # Agentic loop — keep calling tools until Claude is done
    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result,
                })

        # Continue conversation with tool results
        messages = messages + [
            {"role": "assistant", "content": response.content},
            {"role": "user",      "content": tool_results},
        ]
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

    # Extract final text response
    return "".join(
        block.text for block in response.content if hasattr(block, "text")
    )


# ── Streamlit UI ───────────────────────────────
def main():
    st.set_page_config(
        page_title="CropBalanceAI",
        page_icon="🌾",
        layout="centered",
    )

    st.title("🌾 CropBalanceAI")
    st.caption("AI assistant for Indian farmers | भारतीय किसानों के लिए AI सहायक")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        default_state = st.selectbox("Your State / राज्य", [
            "Punjab", "Maharashtra", "Uttar Pradesh", "Madhya Pradesh",
            "Gujarat", "Haryana", "Rajasthan", "Karnataka",
            "Andhra Pradesh", "Telangana", "West Bengal",
        ])
        st.markdown("---")
        st.markdown("**Example questions:**")
        st.markdown("- What is wheat price in Punjab?")
        st.markdown("- गेहूं का भाव क्या होगा?")
        st.markdown("- Should I sell my tomatoes now?")
        st.markdown("- Upload a leaf photo to detect disease")
        st.markdown("---")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Welcome message
        st.session_state.messages.append({
            "role":    "assistant",
            "content": f"Namaste! 🌾 I am CropBalanceAI. I can help you with crop prices, disease detection, and farming advice. You are in **{default_state}**. Ask me anything! \n\nनमस्ते! मैं आपकी फसल की कीमतों, बीमारियों और खेती की जानकारी में मदद कर सकता हूं।",
        })

    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Image upload
    uploaded = st.file_uploader(
        "Upload leaf photo for disease detection 🌿",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    # Chat input
    if prompt := st.chat_input("Ask about crop prices or diseases..."):
        # Add state context to prompt
        full_prompt = f"[My state is {default_state}] {prompt}"

        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
            if uploaded:
                st.image(uploaded, width=200)

        st.session_state.messages.append({"role": "user", "content": full_prompt})

        # Get image if uploaded
        image_b64 = None
        if uploaded:
            image_b64 = base64.b64encode(uploaded.read()).decode()
            if not full_prompt.strip().endswith(("?", ".")):
                full_prompt += " Please detect the disease in this leaf."
            st.session_state.messages[-1]["content"] = full_prompt

        # Get Claude response
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🌾"):
                # Build message history for Claude
                claude_messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                    if m["role"] in ["user", "assistant"]
                ]

                reply = chat_with_claude(claude_messages, image_b64)
                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()