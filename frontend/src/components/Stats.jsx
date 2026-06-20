import {
  FaSeedling,
  FaMapMarkerAlt,
  FaRobot,
  FaChartLine,
} from "react-icons/fa";

export default function Stats() {
  const stats = [
    {
      icon: <FaSeedling />,
      value: "150K+",
      title: "AI Predictions",
    },
    {
      icon: <FaMapMarkerAlt />,
      value: "28",
      title: "States Covered",
    },
    {
      icon: <FaRobot />,
      value: "25+",
      title: "Crop Categories",
    },
    {
      icon: <FaChartLine />,
      value: "95%",
      title: "Prediction Accuracy",
    },
  ];

  return (
    <section className="stats-section">
      <h2>Trusted Across India</h2>

      <div className="stats-grid">
        {stats.map((item, index) => (
          <div className="stat-card" key={index}>
            <div className="stat-icon">{item.icon}</div>

            <h1>{item.value}</h1>

            <p>{item.title}</p>
          </div>
        ))}
      </div>
    </section>
  );
}