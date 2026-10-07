from agent.context.models import FarmerContext


class ContextStore:
    def __init__(self):
        self._contexts: dict[str, FarmerContext] = {}

    def save(self, context: FarmerContext) -> None:
        self._contexts[context.farmer_id] = context

    def get(self, farmer_id: str) -> FarmerContext | None:
        return self._contexts.get(farmer_id)

    def update(self, farmer_id: str, **updates) -> FarmerContext:
        context = self._contexts.get(farmer_id)

        if context is None:
            raise KeyError(f"Farmer context not found: {farmer_id}")

        for field, value in updates.items():
            if value is not None and hasattr(context, field):
                setattr(context, field, value)

        return context