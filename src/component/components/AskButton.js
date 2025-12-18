import React, { useState } from "react";

export default function AskButton() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const getSelectedText = () => {
    if (window.getSelection) {
      return window.getSelection().toString();
    }
    return "";
  };

  const askQuestion = async () => {
    const selectedText = getSelectedText();
    if (!question) {
      alert("Please type your question!");
      return;
    }

    const response = await fetch("/api/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question, selected_text: selectedText }),
    });

    const data = await response.json();
    setAnswer(data.answer);
  };

  return (
    <div style={{ marginTop: "20px" }}>
      <input
        type="text"
        placeholder="Type your question..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        style={{ width: "60%", padding: "8px", marginRight: "10px" }}
      />
      <button onClick={askQuestion} style={{ padding: "8px 15px" }}>
        Ask Book
      </button>
      {answer && (
        <div style={{ marginTop: "15px", padding: "10px", background: "#f0f0f0" }}>
          <strong>Answer:</strong> {answer}
        </div>
      )}
    </div>
  );
}
