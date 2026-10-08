from agent.agent import CropBalanceAgent

agent1 = CropBalanceAgent()

result1 = agent1.run(
    request="What is the price of my crop?",
    farmer_id="farmer_ramu",
)

print("AGENT 1:")
print(result1["parsed"])


agent2 = CropBalanceAgent()

result2 = agent2.run(
    request="What is the price of my crop?",
    farmer_id="farmer_ramu",
)

print("AGENT 2:")
print(result2["parsed"])