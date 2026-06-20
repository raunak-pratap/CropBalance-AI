import React from "react";

export default function Navbar() {
  return (
    <nav className="navbar">
      <h2>🌾 CropBalance AI</h2>

      <div className="nav-links">
        <a href="/">Home</a>
        <a href="/prediction">Dashboard</a>
        <a href="/">Weather</a>
        <a href="/">AI Assistant</a>
      </div>
    </nav>
  );
}