from datetime import datetime

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import DateTime

class Base(DeclarativeBase):
    pass



class Farmer(Base):
    __tablename__ = "farmers"

    farmer_id: Mapped[str] = mapped_column(primary_key=True)
    state: Mapped[str | None] = mapped_column(nullable=True)
    district: Mapped[str | None] = mapped_column(nullable=True)
    current_crop: Mapped[str | None] = mapped_column(nullable=True)
    land_size: Mapped[float | None] = mapped_column(nullable=True)
    soil_type: Mapped[str | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
        nullable=False,
    )