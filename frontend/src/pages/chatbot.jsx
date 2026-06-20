import { useState } from "react";

function Chatbot() {
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState([]);

  const sendMessage = async () => {
    if (!message) return;

    const userMessage = { sender: "user", text: message };
    setChat((prev) => [...prev, userMessage]);

    try {
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message,
          crop: "wheat",
          state: "Punjab"
        })
      });

      const data = await res.json();

      const botMessage = {
        sender: "bot",
        text: data.response
      };

      setChat((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error(error);
    }

    setMessage("");
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>CropBalance AI Chatbot</h1>

      <div style={{
        height: "400px",
        overflowY: "auto",
        border: "1px solid gray",
        padding: "10px",
        marginBottom: "20px"
      }}>
        {chat.map((msg, index) => (
          <div key={index}>
            <strong>{msg.sender}: </strong> {msg.text}
          </div>
        ))}
      </div>

      <input
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Ask about crops..."
        style={{ width: "70%", padding: "10px" }}
      />

      <button onClick={sendMessage}>
        Send
      </button>
    </div>
  );
}

export default Chatbot;