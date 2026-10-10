
from uuid import uuid4

from agent.context.models import FarmerContext
from agent.context.postgres_memory import PostgresContextStore


def test_farmer_context_can_be_saved_and_retrieved():
    store = PostgresContextStore()
    farmer_id = f"pytest_{uuid4().hex}"

    context = FarmerContext(
        farmer_id=farmer_id,
        state="Karnataka",
        district="Bangalore",
        current_crop="Wheat",
        land_size=2.5,
        soil_type="Loamy",
    )

    try:
        store.save(context)
        saved = store.get(farmer_id)

        assert saved is not None
        assert saved.farmer_id == farmer_id
        assert saved.state == "Karnataka"
        assert saved.district == "Bangalore"
        assert saved.current_crop == "Wheat"
        assert saved.land_size == 2.5
        assert saved.soil_type == "Loamy"
    finally:
        store.delete(farmer_id)



def test_partial_update_preserves_other_fields():
    store = PostgresContextStore()
    farmer_id = f"pytest_{uuid4().hex}"

    context = FarmerContext(
        farmer_id=farmer_id,
        state="Maharashtra",
        district="Pune",
        current_crop="Rice",
        land_size=2.5,
        soil_type="Loamy",
    )

    try:
        store.save(context)

        updated = store.update(
            farmer_id=farmer_id,
            current_crop="Wheat",
        )

        assert updated.current_crop == "Wheat"
        assert updated.state == "Maharashtra"
        assert updated.district == "Pune"
        assert updated.land_size == 2.5
        assert updated.soil_type == "Loamy"

        # Verify the persisted database record too.
        saved = store.get(farmer_id)
        assert saved is not None
        assert saved.current_crop == "Wheat"
        assert saved.state == "Maharashtra"
        assert saved.district == "Pune"
    finally:
        store.delete(farmer_id)
