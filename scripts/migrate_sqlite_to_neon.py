import sqlite3, os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
sqlite_conn = sqlite3.connect("vies_detector.db")
pg_engine = create_engine(os.environ["DATABASE_URL"])

tables = [r[0] for r in sqlite_conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()]

for t in tables:
    print(f"Migrating {t}...")
    df = pd.read_sql(f"SELECT * FROM {t}", sqlite_conn)
    df.to_sql(t, pg_engine, if_exists="replace", index=False)
    print(f"  ✓ {len(df)} rows")
