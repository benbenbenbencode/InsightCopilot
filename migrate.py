import pandas as pd
from requests.compat import quote_plus
from sqlalchemy import create_engine




password = quote_plus("RetailDB@2026!")

# Supabase
engine = create_engine(
    "postgresql://postgres.zprntahvfjuozyqoqrhb:{password}@aws-1-ap-northeast-2.pooler.supabase.com:5432/postgres"
)

# Local
local_engine = create_engine(
    "postgresql://postgres:admin@localhost:5432/retail"
)

tables = ["daily_transactions", "user_summary", "product_metrics"]

for table in tables:
    print(f"Copying {table}...")
    df = pd.read_sql(f"SELECT * FROM {table}", local_engine)
    df.to_sql(table, engine, if_exists="replace", index=False)
    print(f"✅ {table} done — {len(df)} rows")

print("All done!")