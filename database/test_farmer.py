from database.connection import SessionLocal
from database.models import Farmer

session = SessionLocal()

saved_farmer = session.get(Farmer, "farmer_001")
saved_farmer.current_crop = "onion"

session.commit()

print(saved_farmer.current_crop)
print(saved_farmer.updated_at)

print(saved_farmer.farmer_id)
print(saved_farmer.state)
print(saved_farmer.district)
print(saved_farmer.current_crop)
print(saved_farmer.land_size)
print(saved_farmer.soil_type)
