from database.connection import SessionLocal
from database.models import Farmer

session = SessionLocal()

try:
    saved_farmer = session.get(Farmer, "farmer_001")

    if saved_farmer is None:
        print("Farmer not found.")
    else:
        print("Farmer ID:", saved_farmer.farmer_id)
        print("State:", saved_farmer.state)
        print("District:", saved_farmer.district)
        print("Current crop:", saved_farmer.current_crop)
        print("Land size:", saved_farmer.land_size)
        print("Soil type:", saved_farmer.soil_type)
        print("Updated at:", saved_farmer.updated_at)

finally:
    session.close()