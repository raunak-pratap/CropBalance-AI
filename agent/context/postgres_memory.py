from agent.context.models import FarmerContext
from database.connection import SessionLocal
from database.models import Farmer

class PostgresContextStore:

    def save(self, context: FarmerContext) -> None:
        session = SessionLocal()
        farmer = session.get(Farmer, context.farmer_id)

        if farmer is None:
            farmer = Farmer(
                farmer_id=context.farmer_id,
                state=context.state,
                current_crop=context.current_crop,
                district=context.district,
                land_size=context.land_size,
                soil_type=context.soil_type,
            )
            session.add(farmer)
        else:
            farmer.state = context.state
            farmer.district = context.district
            farmer.current_crop = context.current_crop
            farmer.land_size = context.land_size
            farmer.soil_type = context.soil_type

        

        session.add(farmer)
        session.commit()
        session.close()

    def get(self, farmer_id: str) -> FarmerContext | None:
        session = SessionLocal()
        farmer = session.get(Farmer, farmer_id)
        session.close()

        if farmer:
            return FarmerContext(
                farmer_id=farmer.farmer_id,
                state=farmer.state,
                district=farmer.district,
                current_crop=farmer.current_crop,
                land_size=farmer.land_size,
                soil_type=farmer.soil_type,
            )
        else:
            return None


    def update(
        self,
        farmer_id: str,
        **updates,
    ) -> FarmerContext:
        session = SessionLocal()
        farmer = session.get(Farmer, farmer_id)

        if not farmer:
            session.close()
            raise ValueError(f"Farmer with ID {farmer_id} not found.")

        allowed_fields = {
            "state",
            "district",
            "current_crop",
            "land_size",
            "soil_type",
        }

        for key, value in updates.items():
            if key not in allowed_fields:
                session.close()
                raise ValueError(f"Invalid attribute: {key}")

            if value is not None:
                setattr(farmer, key, value)

        session.commit()
        updated_context = FarmerContext(
            farmer_id=farmer.farmer_id,
            state=farmer.state,
            district=farmer.district,
            current_crop=farmer.current_crop,
            land_size=farmer.land_size,
            soil_type=farmer.soil_type,
        )
        session.close()
        return updated_context

    def delete(self, farmer_id: str) -> None:
        """Delete a farmer record by ID."""
        session = SessionLocal()
        try:
            farmer = session.get(Farmer, farmer_id)
            if farmer is not None:
                session.delete(farmer)
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
