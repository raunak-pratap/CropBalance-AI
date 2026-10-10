
from agent.context.memory import ContextStore
from agent.context.models import FarmerContext


def test_farmer_context_can_be_saved_and_retrieved():
    store = ContextStore()

    farmer = FarmerContext(
        farmer_id="test_farmer",
        state="Maharashtra",
        current_crop="Wheat",
    )

    store.save(farmer)

    result = store.get("test_farmer")

    assert result is not None
    assert result.farmer_id == "test_farmer"
    assert result.state == "Maharashtra"
    assert result.current_crop == "Wheat"
