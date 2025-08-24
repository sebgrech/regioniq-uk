# scripts/clean/population_nuts1.py
from pathlib import Path
import pandas as pd

IN = Path("data/raw/population_nuts1_nomis.csv")
OUT_DIR = Path("data/clean"); OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LONG = OUT_DIR / "population_ITL1_metrics_long.csv"
OUT_WIDE = OUT_DIR / "population_ITL1_metrics_wide.csv"

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

# Name/code hygiene
df["region"] = df["region"].str.replace("Yorkshire and The Humber", "Yorkshire & Humber", regex=False)
df["region_code"] = df["region_code"].str.strip().str.upper()
df = df.drop_duplicates()

# --- Extract two key series: All ages + Working-age (16–64) ---
all_ages = df[df["age_band"].str.lower() == "all ages"].copy()
working_age = df[df["age_band"].str.contains("16 - 64", case=False)].copy()

if all_ages.empty or working_age.empty:
    raise ValueError("Missing required 'All ages' or 'Aged 16-64' rows. Check raw file.")

# Label metrics
all_ages["metric"] = "total_population"
working_age["metric"] = "working_age_population"

# Combine into one tidy long dataframe
metrics_long = pd.concat([all_ages, working_age], ignore_index=True)
metrics_long = (
    metrics_long[["region", "region_code", "year", "metric", "value"]]
    .dropna(subset=["year", "value"])
    .sort_values(["region_code", "metric", "year"])
    .reset_index(drop=True)
)

metrics_long.to_csv(OUT_LONG, index=False)

# --- Pivot to wide format (regions × years, one metric per block) ---
metrics_wide = (
    metrics_long
    .pivot_table(index=["region", "region_code", "metric"], columns="year", values="value", aggfunc="sum")
    .reset_index()
)

metrics_wide.columns.name = None

# Reorder for neatness
id_cols = ["region", "region_code", "metric"]
year_cols = sorted([c for c in metrics_wide.columns if c not in id_cols])
metrics_wide = metrics_wide[id_cols + year_cols].sort_values(["region_code", "metric"]).reset_index(drop=True)

metrics_wide.to_csv(OUT_WIDE, index=False)

print(f"💾 Saved tidy long → {OUT_LONG}")
print(f"💾 Saved wide (multi-metric, forecast-ready) → {OUT_WIDE}")

print("\n✅ QA:")
print("Metrics included:", metrics_wide["metric"].unique().tolist())
print("Year span:", (min(year_cols) if year_cols else None), "→", (max(year_cols) if year_cols else None))
print("Regions:", metrics_wide['region_code'].nunique())
print("Rows (region × metric):", metrics_wide.shape[0])
print("Nulls in wide:", int(metrics_wide[year_cols].isna().sum().sum()))
