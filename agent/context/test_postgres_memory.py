from agent.context.models import FarmerContext
from agent.context.postgres_memory import PostgresContextStore

store = PostgresContextStore()

context = FarmerContext(
    farmer_id="farmer_002",
    state="Maharashtra",
    district="Pune",
    current_crop="wheat",
    land_size=3.0,
    soil_type="black_soil",
)

store.save(context)