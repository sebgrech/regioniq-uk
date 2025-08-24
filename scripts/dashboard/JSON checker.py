
import json
import pandas as pd
from pathlib import Path

# --- Paths ---
GEO_ITL1_PATH = Path("data/geo/uk_itl1_2025_bgc.geojson")
FORECAST_LONG_PATH = Path("data/forecast/population_ITL1_hist_forecast_long.csv")

# --- Load GeoJSON ---
with open(GEO_ITL1_PATH) as f:
    gj = json.load(f)

print("Top-level keys:", list(gj.keys()))
print("Number of features:", len(gj["features"]))

first = gj["features"][0]
print("First feature properties keys:", list(first["properties"].keys()))
print("First feature geometry type:", first["geometry"]["type"])
print("Example ITL code property value:", first["properties"].get("ITL125CD"))

# Collect ITL codes from geojson
itl_codes = [f["properties"].get("ITL125CD") for f in gj["features"] if "ITL125CD" in f["properties"]]
geojson_codes = sorted(set(itl_codes))

print("Number of ITL codes found:", len(itl_codes))
print("Unique ITL codes:", geojson_codes)

# --- Load forecast_long.csv ---
try:
    df_long = pd.read_csv(FORECAST_LONG_PATH)
    print("\nForecast data loaded:", df_long.shape)

    # Try to detect the region code column
    possible_cols = [c for c in df_long.columns if "code" in c.lower() or "itl" in c.lower()]
    if not possible_cols:
        raise ValueError(f"⚠️ No obvious code column found. Columns are: {df_long.columns}")

    code_col = possible_cols[0]   # pick the first candidate
    print(f"Using column '{code_col}' as region code.")

    df_codes = sorted(df_long[code_col].unique())
    print("Unique ITL codes in df_long:", df_codes)

    # Compare sets
    missing_in_df = set(geojson_codes) - set(df_codes)
    missing_in_geo = set(df_codes) - set(geojson_codes)

    print("✅ Codes in both:", set(df_codes) & set(geojson_codes))
    print("⚠️ Missing in df_long (present in GeoJSON only):", missing_in_df)
    print("⚠️ Missing in GeoJSON (present in df_long only):", missing_in_geo)

except Exception as e:
    print("\n⚠️ Could not load forecast_long.csv:", e)
