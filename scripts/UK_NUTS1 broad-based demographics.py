import os
import urllib.request
import pandas as pd

# === FOLDER SETUP ===
os.makedirs("raw", exist_ok=True)
os.makedirs("data/clean", exist_ok=True)

# === INGEST ===
url = "https://www.nomisweb.co.uk/api/v01/dataset/NM_31_1.data.csv?geography=2013265921...2013265932&sex=7&age=0,24,22,25,20,21&measures=20100"
raw_path = "raw/raw_population.csv"
clean_wide_path = "data/clean/population_wide.csv"

urllib.request.urlretrieve(url, raw_path)
df = pd.read_csv(raw_path)

# === CLEANING ===
columns_to_keep = ['DATE', 'GEOGRAPHY_NAME', 'GEOGRAPHY_CODE', 'AGE_NAME', 'OBS_VALUE']
df = df[columns_to_keep].copy()
df.columns = ['Year', 'Region', 'Region_Code', 'Age_Band', 'Value']
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df['Region'] = df['Region'].str.replace("Yorkshire and The Humber", "Yorkshire & Humber")

# === PIVOT TO WIDE FORMAT ===
df_wide = df.pivot_table(
    index=["Region", "Region_Code", "Age_Band"],
    columns="Year",
    values="Value"
).reset_index()

df_wide.columns.name = None
# Separate ID and year columns
id_cols = ['Region', 'Region_Code', 'Age_Band']
year_cols = sorted([col for col in df_wide.columns if col not in id_cols])

# Reorder the columns: IDs first, then years in order
df_wide = df_wide[id_cols + year_cols]


# === EXPORT ===
df_wide.to_csv(clean_wide_path, index=False)
print(f"✅ Wide-format population data saved to {clean_wide_path}")
