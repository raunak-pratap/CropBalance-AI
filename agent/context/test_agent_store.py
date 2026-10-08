from agent.agent import CropBalanceAgent
from agent.context.memory import ContextStore
from agent.context.models import FarmerContext

store = ContextStore()

store.save(
    FarmerContext(
        farmer_id="test_farmer",
        state="Maharashtra",
        current_crop="Wheat",
    )
)

agent = CropBalanceAgent(context_store=store)

result = agent.run(
    request="What is the price of my crop?",
    farmer_id="test_farmer",
)

print(result["parsed"])