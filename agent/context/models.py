from dataclasses import dataclass


@dataclass
class FarmerContext:
    farmer_id: str
    state: str | None = None
    district: str | None = None
    current_crop: str | None = None
    land_size: float | None = None
    soil_type: str | None = None