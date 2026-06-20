import streamlit as st
import anthropic
import requests
import base64
import re

API_BASE = "http://localhost:8000"
CLAUDE_MODEL = "claude-sonnet-4-5"
client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "predict_crop_price",
        "description": "Predict future crop prices",
        "input_schema": {
            "type": "object",
            "properties": {
                "crop": {"type": "string"},
                "state": {"type": "string"}
            },
            "required": ["crop", "state"]
        }
    },
    {
        "name": "get_live_price",
        "description": "Get current mandi price",
        "input_schema": {
            "type": "object",
            "properties": {
                "crop": {"type": "string"},
                "state": {"type": "string"}
            },
            "required": ["crop", "state"]
        }
    },
    {
        "name": "detect_disease",
        "description": "Detect disease from crop image",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_base64": {"type": "string"}
            },
            "required": ["image_base64"]
        }
    },
    {
        "name": "farming_advice",
        "description": "Provide crop farming advice",
        "input_schema": {
            "type": "object",
            "properties": {
                "crop": {"type": "string"},
                "question": {"type": "string"}
            },
            "required": ["crop", "question"]
        }
    }
]

SYSTEM_PROMPT = """
You are CropBalanceAI, an intelligent AI farming assistant.

You help farmers with:
- Crop prices
- Crop prediction
- Disease detection
- Farming advice
- Selling decisions

Rules:
- Reply in same language as user.
- Be conversational like ChatGPT.
- Use state mentioned by user.
- Maintain conversation memory.
- If missing crop/state ask follow-up question.
"""

def execute_tool(tool_name, tool_input):
    try:
        if tool_name == "predict_crop_price":
            resp = requests.post(
                f"{API_BASE}/predict",
                json={
                    "crop": tool_input["crop"],
                    "state": tool_input["state"],
                    "days_history": 90
                }
            )
            data = resp.json()
            forecast = data["forecast"]
            avg = sum(x["price_inr"] for x in forecast) / len(forecast)
            return f"Predicted average price: ₹{avg:.2f}/quintal"

        elif tool_name == "get_live_price":
            resp = requests.get(
                f"{API_BASE}/prices/live",
                params={
                    "crop": tool_input["crop"],
                    "state": tool_input["state"]
                }
            )
            data = resp.json()
            return f"Current {data['crop']} price in {data['state']} is ₹{data['modal_price_inr']} per quintal."

        elif tool_name == "detect_disease":
            image_bytes = base64.b64decode(tool_input["image_base64"])
            resp = requests.post(
                f"{API_BASE}/disease/detect",
                files={"image": ("leaf.jpg", image_bytes, "image/jpeg")}
            )
            data = resp.json()
            return f"""
Disease: {data['disease']}
Confidence: {data['confidence']*100:.2f}%
Severity: {data['severity']}
Treatment: {data['treatment_en']}
"""

        elif tool_name == "farming_advice":
            crop = tool_input["crop"].lower()

            if crop == "wheat":
                return """
Wheat Farming Advice:
- Sowing: Oct-Dec
- Harvest: Mar-Apr
- Best Temp: 10-25°C
- Soil: Loamy soil
- Water: 4-5 irrigations
"""

            elif crop == "tomato":
                return """
Tomato Farming Advice:
- Sowing: Sep-Oct / Jan-Feb
- Temp: 20-27°C
- Soil: Well-drained loamy soil
- Water: Regular watering
"""

            return "General advice: Maintain good irrigation, soil nutrition, and pest monitoring."

    except Exception as e:
        return str(e)

def chat_with_claude(messages, image_b64=None):
    if image_b64:
        messages[-1]["content"] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64
                }
            },
            {
                "type": "text",
                "text": "Analyze this crop leaf"
            }
        ]

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages
    )

    while response.stop_reason == "tool_use":
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages = messages + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results}
        ]

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

    return "".join(
        block.text for block in response.content if hasattr(block, "text")
    )

def main():
    st.set_page_config(page_title="CropBalanceAI", page_icon="🌾")
    st.title("🌾 CropBalance AI")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    uploaded = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        image_b64 = None
        if uploaded:
            image_b64 = base64.b64encode(uploaded.read()).decode()

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = chat_with_claude(st.session_state.messages, image_b64)
                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()