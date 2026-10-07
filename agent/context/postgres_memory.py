from agent.context.models import FarmerContext
from database.connection import SessionLocal
from database.models import Farmer

class PostgresContextStore:

    def save(self, context: FarmerContext) -> None:
        session = SessionLocal()

        farmer = Farmer(
            farmer_id=context.farmer_id,
            state=context.state,
            current_crop=context.current_crop,
            district=context.district,
            land_size=context.land_size,
            soil_type=context.soil_type,
        )

        session.add(farmer)
        session.commit()
        session.close()