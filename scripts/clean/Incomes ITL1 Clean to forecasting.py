# scripts/clean/income_gdhi_itl1.py
from pathlib import Path
import pandas as pd

IN = Path("data/raw/incomes/gdhi_itl1_nomis.csv")  # must match your ingest output
OUT_DIR = Path("data/clean/incomes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LONG = OUT_DIR / "income_GDHI_ITL1_long.csv"
OUT_WIDE = OUT_DIR / "income_GDHI_ITL1_wide.csv"

if not IN.exists():
    raise FileNotFoundError(
        "Raw file missing. Run scripts/ingest/income_gdhi_nomis_itl1.py first."
    )

# --- Load & select lean columns ---pwd
df = pd.read_csv(IN)

keep = [
    "DATE",
    "GEOGRAPHY_NAME",
    "GEOGRAPHY_CODE",
    "COMPONENT_OF_GDHI_NAME",
    "MEASURE",        # numeric code (1 = £m, 2 = per head £)
    "OBS_VALUE",
]
missing = [c for c in keep if c not in df.columns]
if missing:
    raise KeyError(f"Expected columns not found in raw CSV: {missing}")

df = df[keep].copy()
df.columns = ["year", "region", "region_code", "component", "measure_code", "value"]

# --- Canonical tidy-ups ---
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["measure_code"] = pd.to_numeric(df["measure_code"], errors="coerce").astype("Int64")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# Region name/code hygiene (match your population cleaner)
df["region"] = (
    df["region"]
    .str.replace("Yorkshire and The Humber", "Yorkshire & Humber", regex=False)
    .str.strip()
)
df["region_code"] = df["region_code"].str.strip().str.upper()

df = df.dropna(subset=["year", "measure_code", "value"]).drop_duplicates()

# --- Sanity check: only total GDHI component present (ingest used component_of_gdhi=0) ---
expected_component = "Gross Disposable Household Income (GDHI)"
if df["component"].nunique() != 1 or expected_component not in df["component"].unique():
    raise ValueError(
        "Expected only total GDHI rows. Check ingest params (component_of_gdhi=0)."
    )

# --- Map MEASURE code to canonical metric keys (avoid brittle unicode names) ---
metric_map = {
    1: "gdhi_total_mn_gbp",   # GDHI (£m)
    2: "gdhi_per_head_gbp",   # GDHI per head (£)
}
df["metric"] = df["measure_code"].map(metric_map)
df = df.dropna(subset=["metric"])

# --- Long format (aligns with your population schema) ---
long_df = (
    df[["region", "region_code", "year", "metric", "value"]]
    .sort_values(["region_code", "metric", "year"])
    .reset_index(drop=True)
)
long_df.to_csv(OUT_LONG, index=False)

# --- Wide format (regions × years, one row per region×metric) ---
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
print("Metrics:", wide_df["metric"].unique().tolist())
print("Year span:", (min(year_cols) if year_cols else None), "→", (max(year_cols) if year_cols else None))
print("Regions:", wide_df["region_code"].nunique())
print("Nulls in wide:", int(pd.isna(wide_df[year_cols]).sum().sum()))
