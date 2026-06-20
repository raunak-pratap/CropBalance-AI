import { useNavigate } from "react-router-dom";

export default function Hero() {
  const navigate = useNavigate();

  return (
    <section className="hero">
      <h1>🌾 CropBalance AI</h1>
      <p>
        Predict crop prices using Machine Learning, market intelligence, and AI
        recommendations for smarter farming decisions.
      </p>

      <div className="hero-buttons">
        <button
          className="primary-btn"
          onClick={() => navigate("/prediction")}
        >
          🚀 Try Live Demo
        </button>

        <button
          className="secondary-btn"
          onClick={() => navigate("/dashboard")}
        >
          📈 View Dashboard
        </button>
      </div>
    </section>
  );
}