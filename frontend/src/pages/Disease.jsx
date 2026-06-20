import React, { useState } from "react";
import Navbar from "../components/Navbar";

export default function Disease() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!image) {
      alert("Please select an image");
      return;
    }

    const formData = new FormData();
    formData.append("image", image);

    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/disease/detect", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      console.log(data);
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Disease detection failed");
    }

    setLoading(false);
  };

  return (
    <>
      <Navbar />

      <div className="prediction-container">
        <h1>Disease Detection</h1>

        <div style={{ marginTop: "30px" }}>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImage(e.target.files[0])}
          />

          <br /><br />

          <button className="predict-btn" onClick={handleUpload}>
            {loading ? "Detecting..." : "Detect Disease"}
          </button>
        </div>

        {result && (
          <div style={{ marginTop: "40px" }}>
            <h2>Detection Result</h2>

            <div className="card">
              <h3>Disease</h3>
              <p>{result.disease}</p>
            </div>

            <div className="card">
              <h3>Confidence</h3>
              <p>{(result.confidence * 100).toFixed(2)}%</p>
            </div>

            <div className="card">
              <h3>Treatment</h3>
              <p>{result.treatment_en}</p>
            </div>
          </div>
        )}
      </div>
    </>
  );
}