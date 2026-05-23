
import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_raw_database_url = os.getenv("DATABASE_URL", "")

if _raw_database_url.startswith("postgres://"):
    # SQLAlchemy 2 prefers the modern URI scheme for PostgreSQL
    _raw_database_url = _raw_database_url.replace("postgres://", "postgresql://", 1)

if _raw_database_url.startswith("sqlite:///"):
    sqlite_path = _raw_database_url[len("sqlite:///") :]
    if sqlite_path.startswith("./") or sqlite_path == "idrms.db" or not os.path.isabs(sqlite_path):
        sqlite_path = os.path.normpath(os.path.join(BASE_DIR, sqlite_path))
    DATABASE_URL = f"sqlite:///{sqlite_path}"
elif _raw_database_url:
    DATABASE_URL = _raw_database_url
else:
    DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'idrms.db')}"

# SQLite needs connect_args; remove the kwarg for other engines
connect_kwargs = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency – yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Called once at startup to create all tables that don't exist yet."""
    # Import all models so Base.metadata knows about them
    import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
