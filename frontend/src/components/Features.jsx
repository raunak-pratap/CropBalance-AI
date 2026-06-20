export default function Features() {
    const items = [
      {
        icon: "📈",
        title: "Price Forecast",
        text: "Predict future crop prices using AI.",
      },
      {
        icon: "🌦",
        title: "Weather",
        text: "Weather intelligence for farmers.",
      },
      {
        icon: "🤖",
        title: "AI Assistant",
        text: "Get AI recommendations instantly.",
      },
      {
        icon: "💰",
        title: "Profit Calculator",
        text: "Estimate expected earnings.",
      },
    ];
  
    return (
      <section className="features">
  
        <h2>Why CropBalance AI?</h2>
  
        <div className="feature-grid">
  
          {items.map((item, index) => (
            <div
              className="feature-card"
              key={index}
            >
              <h1>{item.icon}</h1>
  
              <h3>{item.title}</h3>
  
              <p>{item.text}</p>
            </div>
          ))}
  
        </div>
  
      </section>
    );
  }