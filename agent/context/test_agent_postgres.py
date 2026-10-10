
from uuid import uuid4

from agent.agent import CropBalanceAgent
from agent.context.models import FarmerContext
from agent.context.postgres_memory import PostgresContextStore


def test_agent_uses_saved_farmer_context():
    store = PostgresContextStore()
    farmer_id = f"pytest_{uuid4().hex}"

    context = FarmerContext(
        farmer_id=farmer_id,
        state="Maharashtra",
        district="Pune",
        current_crop="Wheat",
        land_size=2.5,
        soil_type="Loamy",
    )

    try:
        store.save(context)

        agent = CropBalanceAgent(context_store=store)

        result = agent.run(
            request="What is the price of my crop?",
            farmer_id=farmer_id,
        )

        assert result["parsed"]["intent"] == "price_prediction"
        assert result["parsed"]["crop"] == "Wheat"
        assert result["parsed"]["state"] == "Maharashtra"

    finally:
        store.delete(farmer_id)
