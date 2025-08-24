# scripts/clean/emp/emp_itl1.py
from pathlib import Path
import pandas as pd

IN = Path("data/raw/emp/emp_itl1_nomis.csv")  # merged 2009–2015 + 2015–2023
OUT_DIR = Path("data/clean/emp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LONG = OUT_DIR / "emp_ITL1_long.csv"
OUT_WIDE = OUT_DIR / "emp_ITL1_wide.csv"

if not IN.exists():
    raise FileNotFoundError("Raw file missing. Run scripts/ingest/emp/emp_itl1_nomis.py first.")

# --- Load raw NOMIS CSV ---
df = pd.read_csv(IN)

# Keep a lean set of columns; others are metadata we don’t need downstream
keep = [
    "DATE",             # year
    "GEOGRAPHY_NAME",   # ITL1 region name (NOMIS)
    "GEOGRAPHY_CODE",   # ITL1 region code (E12..., W92..., S92..., N92...)
    "INDUSTRY_NAME",    # should be "All industries"
    "EMPLOYMENT_STATUS_NAME",  # should be "Total"
    "MEASURES_NAME",    # "Value"
    "OBS_VALUE",        # numeric jobs count
]
missing = [c for c in keep if c not in df.columns]
if missing:
    raise KeyError(f"Expected columns not found in raw CSV: {missing}")

df = df[keep].copy()
df.columns = ["year", "region_nom", "code_nom", "industry", "emp_status", "measure_name", "value"]

# --- Canonical tidy-ups ---
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# Filter to totals only (ingest already requested totals, this is a guardrail)
df = df[df["emp_status"].str.lower().eq("total")]

# Filter to all industries only (ingest used industry=all; guardrail)
df = df[df["industry"].str.contains("all industries", case=False, na=False)]

# Region name/code hygiene (align with GVA ITL1 naming used in your app)
# Map NOMIS regional codes to ONS ITL1 codes (TLC…TLN)
ITL1_CODE_MAP = {
    "E12000001": "TLC",  # North East
    "E12000002": "TLD",  # North West
    "E12000003": "TLE",  # Yorkshire and The Humber
    "E12000004": "TLF",  # East Midlands
    "E12000005": "TLG",  # West Midlands
    "E12000006": "TLH",  # East of England
    "E12000007": "TLI",  # London
    "E12000008": "TLJ",  # South East
    "E12000009": "TLK",  # South West
    "W92000004": "TLL",  # Wales
    "S92000003": "TLM",  # Scotland
    "N92000002": "TLN",  # Northern Ireland
}

# Normalise region names to match your other cleaners
NAME_NORMALISE = {
    "Yorkshire and The Humber": "Yorkshire & Humber",
    "East of England": "East of England",  # placeholder to show pattern
    # add other one-offs here if needed
}

df["region"] = df["region_nom"].replace(NAME_NORMALISE).str.strip()
df["region_code"] = df["code_nom"].map(ITL1_CODE_MAP)

# Drop rows lacking mapping (e.g., UK/England aggregates if ever present)
df = df.dropna(subset=["region_code", "year", "value"])

# If both vintages include the same (region_code, year), prefer the newer dataset
# Our ingest concatenated legacy first then current, so keep='last' retains newer row.
df = df.sort_values(["region_code", "year"]).drop_duplicates(
    subset=["region_code", "year"], keep="last"
)

# Metric key aligned to unified forecasting
df["metric"] = "emp_total_jobs"

# --- Long format (unified schema) ---
long_df = (
    df[["region", "region_code", "year", "metric", "value"]]
    .sort_values(["region_code", "metric", "year"])
    .reset_index(drop=True)
)

long_df.to_csv(OUT_LONG, index=False)

# --- Wide format (rows = region x metric, cols = years) ---
wide_df = (
    long_df.pivot_table(
        index=["region", "region_code", "metric"],
        columns="year",
        values="value",
        aggfunc="first",
    )
    .reset_index()
)
wide_df.columns.name = None

id_cols = ["region", "region_code", "metric"]
year_cols = sorted([c for c in wide_df.columns if c not in id_cols])
wide_df = (
    wide_df[id_cols + year_cols]
    .sort_values(["region_code", "metric"])
    .reset_index(drop=True)
)

wide_df.to_csv(OUT_WIDE, index=False)

# --- QA ---
print(f"💾 Saved tidy long → {OUT_LONG}")
print(f"💾 Saved wide → {OUT_WIDE}")
print("\n✅ QA:")
print("Metric(s):", wide_df["metric"].unique().tolist())
print("Year span:", (min(year_cols) if year_cols else None), "→", (max(year_cols) if year_cols else None))
print("Regions:", wide_df["region_code"].nunique())
print("Nulls in wide:", int(pd.isna(wide_df[year_cols]).sum().sum()))
print("\n🔎 Sample (last 3 years):")
print(wide_df[id_cols + year_cols[-3:]].head(3))
