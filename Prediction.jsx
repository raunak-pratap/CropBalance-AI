import { useState } from "react";

export default function Prediction() {
  const [crop, setCrop] = useState("");
  const [state, setState] = useState("");
  const [result, setResult] = useState(null);

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
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          crop: crop.trim().toLowerCase(),
          state: state.trim(),
          days_history: 90,
        }),
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


  return (
    <div style={{ padding: "50px", textAlign: "center", color: "white" }}>
      <h1>Crop Price Prediction</h1>

      <input
        type="text"
        placeholder="Crop"
        value={crop}
        onChange={(e) => setCrop(e.target.value)}
      />

      <input
        type="text"
        placeholder="State"
        value={state}
        onChange={(e) => setState(e.target.value)}
      />

      <br />

      <button onClick={handlePredict}>Predict</button>

      {result && (
        <div>
          <h2>Prediction Result</h2>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}