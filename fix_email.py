from database import engine
from sqlalchemy import text

with engine.connect() as conn:
    conn.execute(text("ALTER TABLE residents ADD COLUMN email VARCHAR DEFAULT ''"))
    conn.commit()
    print("Done!")