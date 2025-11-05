from urllib.parse import quote_plus
import pandas as pd
from sqlalchemy import create_engine
password = "Kaizen@123#" 
encoded_password = quote_plus(password)
# Load CSV
df = pd.read_csv('data/features/framework_dataset.csv')
# Create connection with encoded password
engine = create_engine(f'mysql+pymysql://root:{encoded_password}@localhost/stocks_analysis')
# Import to SQL
df.to_sql('stocks_data', engine, if_exists='replace', index=False)
print(f"Loaded {len(df)} rows into stocks_data table")