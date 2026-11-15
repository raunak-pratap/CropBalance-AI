"""Create the application's SQLAlchemy tables.

Run explicitly with:
    python -m database.init_db

Importing database.connection itself does not initialize a connection.
"""
from database.connection import get_engine
from database.models import Base


def main() -> None:
    Base.metadata.create_all(bind=get_engine())


if __name__ == "__main__":
    main()
