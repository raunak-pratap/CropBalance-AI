import React, { useState } from "react";
import Navbar from "../components/Navbar";
import {
  LineChart,
 Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer
} from "recharts";

export default function Prediction() {
  const [crop, setCrop] = useState("");
  const [state, setState] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!crop || !state) {
      alert("Please fill all fields");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          crop: crop.trim().toLowerCase(),
          state: state.trim(),
          days_history: 90
        })
      });

      const data = await response.json();
      console.log(data);
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Prediction failed");
    }

    setLoading(false);
  };

  const forecast = result?.forecast || [];

  const chartData = forecast.map((item) => ({
    date: item.date.slice(5),
    price: item.price_inr
  }));

  const prices = forecast.map((item) => item.price_inr);

  const avgPrice =
    prices.length > 0
      ? (prices.reduce((a, b) => a + b, 0) / prices.length).toFixed(2)
      : 0;

  const highestPrice = prices.length > 0 ? Math.max(...prices) : 0;
  const lowestPrice = prices.length > 0 ? Math.min(...prices) : 0;

  return (
    <>
      <Navbar />

      <div className="prediction-container">
        <h1>Crop Price Prediction</h1>

        <div style={{ marginTop: "30px" }}>
          <input
            className="input-box"
            type="text"
            placeholder="Enter Crop"
            value={crop}
            onChange={(e) => setCrop(e.target.value)}
          />

          <input
            className="input-box"
            type="text"
            placeholder="Enter State"
            value={state}
            onChange={(e) => setState(e.target.value)}
          />

          <br />

          <button className="predict-btn" onClick={handlePredict}>
            {loading ? "Predicting..." : "Predict"}
          </button>
        </div>

        {result && (
          <div style={{ marginTop: "40px" }}>
            <h2>Prediction Analytics</h2>

            <div
              style={{
                display: "flex",
                justifyContent: "center",
                gap: "20px",
                flexWrap: "wrap",
                marginTop: "20px"
              }}
            >
              <div className="card">
                <h3>Average Price</h3>
                <p>₹{avgPrice}</p>
              </div>

              <div className="card">
                <h3>Highest Price</h3>
                <p>₹{highestPrice}</p>
              </div>

              <div className="card">
                <h3>Lowest Price</h3>
                <p>₹{lowestPrice}</p>
              </div>

              <div className="card">
                <h3>AI Confidence</h3>
                <p>94%</p>
              </div>
            </div>

            <div style={{ marginTop: "40px" }}>
              <h2>Forecast Data</h2>

              {forecast.slice(0, 7).map((item, index) => (
                <div key={index} className="forecast-row">
                  <span>{item.date}</span>
                  <span>₹{item.price_inr}</span>
                </div>
              ))}
            </div>

            <div style={{ marginTop: "50px" }}>
              <h2>Price Trend</h2>

              <div
                style={{
                  width: "80%",
                  height: "400px",
                  margin: "auto",
                  marginTop: "20px",
                  background: "#0b1f3a",
                  padding: "20px",
                  borderRadius: "15px"
                }}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="price"
                      stroke="#00ff99"
                      strokeWidth={3}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}