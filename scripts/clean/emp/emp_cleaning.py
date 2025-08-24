from pathlib import Path
import pandas as pd

IN = Path("data/raw/emp/emp_itl1_nomis.csv")   # merged 2009–2015 + 2015–2023
OUT_DIR = Path("data/clean/emp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LONG = OUT_DIR / "emp_ITL1_long.csv"
OUT_WIDE = OUT_DIR / "emp_ITL1_wide.csv"

if not IN.exists():
    raise FileNotFoundError("Raw file missing. Run scripts/ingest/emp/emp_itl1_nomis.py first.")

df = pd.read_csv(IN)

# ---- pick canonical columns (NOMIS naming) ----
keep = [
    "DATE", "GEOGRAPHY_NAME", "GEOGRAPHY_CODE",
    "INDUSTRY_NAME", "EMPLOYMENT_STATUS_NAME",
    "MEASURES_NAME", "OBS_VALUE"
]
missing = [c for c in keep if c not in df.columns]
if missing:
    raise KeyError(f"Missing expected columns: {missing}")

df = df[keep].copy()
df.columns = ["year","region_nom","code_nom","industry","emp_status","measure_name","value"]

# ---- types ----
df["year"]  = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# ---- gentle filters that match BRES labels ----
# Industry: BRES returns "Total" for all industries
df = df[df["industry"].str.lower().isin({"total","all industries","all industry"})]

# Employment status: BRES returns "Employees" (employee jobs)
# Accept a few variants so this cleaner also works if you swap in WFJ later.
ok_status = {"employees", "total", "all jobs"}
df = df[df["emp_status"].str.lower().isin(ok_status)]

# ---- map NOMIS region codes to ITL1 (TLC…TLN) used elsewhere ----
ITL1_CODE_MAP = {
    "E12000001":"TLC","E12000002":"TLD","E12000003":"TLE","E12000004":"TLF",
    "E12000005":"TLG","E12000006":"TLH","E12000007":"TLI","E12000008":"TLJ",
    "E12000009":"TLK","W92000004":"TLL","S92000003":"TLM","N92000002":"TLN",
}

NAME_NORMALISE = {
    "Yorkshire and The Humber":"Yorkshire & Humber",
    "East":"East of England",
}

df["region"] = df["region_nom"].replace(NAME_NORMALISE).str.strip()
df["region_code"] = df["code_nom"].map(ITL1_CODE_MAP)

# drop aggregates/unmapped and NAs
df = df.dropna(subset=["region_code","year","value"])

# if both vintages provided the same (region,year), keep the newer (our ingest appended new last)
df = df.sort_values(["region_code","year"]).drop_duplicates(
    subset=["region_code","year"], keep="last"
)

# ---- metric key (keep name you’re using in forecasting) ----
# Note: BRES is employees only; you can rename to 'emp_employee_jobs' later if you want to be precise.
df["metric"] = "emp_total_jobs"

# ---- long format ----
long_df = (
    df[["region","region_code","year","metric","value"]]
      .sort_values(["region_code","metric","year"])
      .reset_index(drop=True)
)

assert len(long_df) > 0, "No rows after cleaning — investigate filters/mappings."
assert long_df["year"].nunique() > 0, "No years found after cleaning."

OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
long_df.to_csv(OUT_LONG, index=False)

# ---- wide format ----
wide_df = (
    long_df.pivot_table(
        index=["region","region_code","metric"],
        columns="year",
        values="value",
        aggfunc="first",
    )
    .reset_index()
)
wide_df.columns.name = None

id_cols = ["region","region_code","metric"]
year_cols = sorted([c for c in wide_df.columns if c not in id_cols])
if not year_cols:
    raise RuntimeError("No year columns produced in pivot — check input and filters.")

wide_df = wide_df[id_cols + year_cols] \
           .sort_values(["region_code","metric"]) \
           .reset_index(drop=True)

wide_df.to_csv(OUT_WIDE, index=False)

# ---- QA ----
print(f"💾 Saved tidy long → {OUT_LONG}")
print(f"💾 Saved wide      → {OUT_WIDE}")
print("\n✅ QA:")
print("Metric(s):", wide_df["metric"].unique().tolist())
print("Year span:", (min(year_cols) if year_cols else None), "→", (max(year_cols) if year_cols else None))
print("Regions:", wide_df["region_code"].nunique())
print("Nulls in wide:", int(pd.isna(wide_df[year_cols]).sum().sum()))
print("\n🔎 Sample (last 3 years):")
print(wide_df[id_cols + year_cols[-3:]].head(3))
