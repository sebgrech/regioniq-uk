# scripts/clean/population_lad.py
from pathlib import Path
import pandas as pd

IN = Path("data/raw/population_LAD_nomis.csv")
OUT_DIR = Path("data/clean"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "population_LAD.csv"

if not IN.exists():
    raise FileNotFoundError("Raw LAD file missing. Run scripts/ingest/population_nomis_LAD.py first.")

df = pd.read_csv(IN)

# --- Select & rename core columns ---
keep = ["DATE", "GEOGRAPHY_NAME", "GEOGRAPHY_CODE", "AGE_NAME", "SEX_NAME", "OBS_VALUE"]
df = df[keep].copy()
df.columns = ["year", "region", "region_code", "age_band", "sex", "value"]

# --- Filter for total population only ---
df = df[df["sex"] == "Total"].drop(columns="sex")

# --- Canonical tidy-ups ---
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df["region_code"] = df["region_code"].str.strip().str.upper()
df = df.drop_duplicates()

# --- Pivot to wide: IDs first, years as columns ---
wide = (
    df.pivot_table(index=["region", "region_code", "age_band"], columns="year", values="value")
      .reset_index()
)

wide.columns.name = None
id_cols = ["region", "region_code", "age_band"]
year_cols = sorted([c for c in wide.columns if c not in id_cols])
wide = wide[id_cols + year_cols]

wide = wide.sort_values(["region_code", "age_band"]).reset_index(drop=True)

wide.to_csv(OUT, index=False)
print(f"💾 Saved clean wide LAD table → {OUT}")


