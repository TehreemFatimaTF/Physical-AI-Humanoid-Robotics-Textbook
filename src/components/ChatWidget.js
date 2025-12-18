import React, { useState } from "react";

export default function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState([]);

  const sendMessage = async () => {
    if (!message) return;

    const userMsg = { role: "user", text: message };
    setChat([...chat, userMsg]);
    setMessage("");

    const selectedText = window.getSelection().toString();

    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: message,
        selected_text: selectedText
      })
    });

    const data = await res.json();

    setChat(prev => [...prev, { role: "bot", text: data.answer }]);
  };

  return (
    <>
      {/* Floating Button */}
      <div
        onClick={() => setOpen(!open)}
        style={{
          position: "fixed",
          right: "20px",
          bottom: "20px",
          background: "#000",
          color: "#fff",
          borderRadius: "50%",
          width: "55px",
          height: "55px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          cursor: "pointer",
          zIndex: 9999
        }}
      >
        💬
      </div>

      {/* Chat Window */}
      {open && (
        <div
          style={{
            position: "fixed",
            right: "20px",
            bottom: "90px",
            width: "320px",
            height: "420px",
            background: "#111",
            color: "#fff",
            borderRadius: "12px",
            padding: "10px",
            display: "flex",
            flexDirection: "column",
            zIndex: 9999
          }}
        >
          <strong>AI Assistant</strong>

          <div style={{ flex: 1, overflowY: "auto", marginTop: "10px" }}>
            {chat.map((c, i) => (
              <div key={i} style={{ marginBottom: "8px" }}>
                <b>{c.role === "user" ? "You" : "AI"}:</b> {c.text}
              </div>
            ))}
          </div>

          <div style={{ display: "flex" }}>
            <input
              value={message}
              onChange={e => setMessage(e.target.value)}
              placeholder="Type message..."
              style={{ flex: 1 }}
            />
            <button onClick={sendMessage}>➤</button>
          </div>
        </div>
      )}
    </>
  );
}
