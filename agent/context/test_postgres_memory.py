from agent.context.postgres_memory import PostgresContextStore
from agent.context.models import FarmerContext

store = PostgresContextStore()

new_context = FarmerContext(
    farmer_id="farmer_ramu",
    state="Karnataka",
    district="Bangalore",
    current_crop="Wheat",
    land_size=2.5,
    soil_type="Loamy",
)

store.save(new_context)

print("Saved context for farmer_ramu.")
print(new_context)