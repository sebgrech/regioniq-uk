# scripts/clean/population_nuts1.py
from pathlib import Path
import pandas as pd

IN = Path("data/raw/population_nuts1_nomis.csv")
OUT_DIR = Path("data/clean"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "population_ITL1_.csv"

if not IN.exists():
    raise FileNotFoundError("Raw file missing. Run scripts/ingest/population_nomis_nuts1.py first.")

df = pd.read_csv(IN)

# --- Select & rename core columns ---
keep = ["DATE", "GEOGRAPHY_NAME", "GEOGRAPHY_CODE", "AGE_NAME", "OBS_VALUE"]
df = df[keep].copy()
df.columns = ["year", "region", "region_code", "age_band", "value"]

# --- Canonical tidy-ups ---
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# Standardise a known name quirk (example)
df["region"] = df["region"].str.replace("Yorkshire and The Humber", "Yorkshire & Humber", regex=False)

# Enforce uppercase NUTS codes & strip whitespace
df["region_code"] = df["region_code"].str.strip().str.upper()

# Drop exact duplicate rows that can arrive via API pagination/joins
df = df.drop_duplicates()

# --- Pivot to wide: IDs first, years as columns ---
wide = (
    df.pivot_table(index=["region", "region_code", "age_band"], columns="year", values="value")
      .reset_index()
)

# Remove the pandas column name for neatness
wide.columns.name = None

# Reorder: id columns then sorted years
id_cols = ["region", "region_code", "age_band"]
year_cols = sorted([c for c in wide.columns if c not in id_cols])
wide = wide[id_cols + year_cols]

# Optional: sort rows by region_code then age_band for stable diffs
wide = wide.sort_values(["region_code", "age_band"]).reset_index(drop=True)

wide.to_csv(OUT, index=False)
print(f"💾 Saved clean wide table → {OUT}")
