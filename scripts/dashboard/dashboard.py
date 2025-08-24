import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path


st.set_page_config(page_title="Region IQ — ITL1 Population", layout="wide")

DATA = Path("data/forecast/population_ITL1_hist_forecast_long.csv")
if not DATA.exists():
    st.error(f"Missing {DATA}. Run the forecast script first.")
    st.stop()

df = pd.read_csv(DATA)

# Ensure schema robustness
for c in ["ci_lower", "ci_upper"]:
    if c not in df.columns:
        df[c] = np.nan

# Types
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
for c in ["ci_lower", "ci_upper"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Sidebar controls
st.sidebar.title("Controls")

# Region select (use code + name)
regions = (
    df[["region_code", "region"]]
    .drop_duplicates()
    .sort_values(["region_code"])
    .assign(label=lambda x: x["region"] + " (" + x["region_code"] + ")")
)

region_label = st.sidebar.selectbox(
    "Region (ITL1)",
    regions["label"].to_list(),
    index=0
)
sel_code = regions.loc[regions["label"] == region_label, "region_code"].iloc[0]
sel_region = regions.loc[regions["label"] == region_label, "region"].iloc[0]

metric_map = {
    "Total population": "total_population",
    "Working-age (16–64)": "working_age_population",
}
metric_label = st.sidebar.radio("Metric", list(metric_map.keys()), index=0, horizontal=False)
sel_metric = metric_map[metric_label]

# Horizon
max_year = int(df["year"].max())
has_2050 = max_year >= 2050
horizon = st.sidebar.radio(
    "Horizon",
    ["2030"] + (["Full"] if has_2050 else []),
    index=0
)
clip_year = 2030 if horizon == "2030" else max_year

# Indexing
index_on = st.sidebar.checkbox("Index to base year = 100", value=False)
# Base year defaults to last historical year for the series; user can change
# We'll set a slider after we know last historical year for the chosen series.

# Fetch series for selected region & metric
series = (
    df[(df["region_code"] == sel_code) & (df["metric"] == sel_metric)]
    .dropna(subset=["year", "value"])
    .sort_values("year")
)

if series.empty:
    st.warning("No data for selection.")
    st.stop()

last_hist_year = int(series.loc[series["source"] == "historical", "year"].max())
min_year = int(series["year"].min())
max_avail_year = int(series["year"].max())
clip_year = min(clip_year, max_avail_year)

# Base year slider
if index_on:
    base_year = st.sidebar.slider("Base year", min_year, last_hist_year, min(last_hist_year, 2015))
else:
    base_year = None

# Build plot-ready frames
hist = series[(series["source"] == "historical") & (series["year"] <= clip_year)].copy()
fcst = series[(series["source"] == "forecast") & (series["year"] <= clip_year)].copy()

def indexify(s: pd.Series, base_y: int):
    if base_y is None or base_y not in s.index:
        return s
    base = s.loc[base_y]
    if pd.isna(base) or base == 0:
        return s
    return (s / base) * 100.0

# Value series
hist_s = hist.set_index("year")["value"]
fcst_s = fcst.set_index("year")["value"]

# Optional indexing
if index_on:
    combined = pd.concat([hist_s, fcst_s])
    combined = indexify(combined, base_year)
    hist_s = combined.loc[hist_s.index]
    fcst_s = combined.loc[fcst_s.index]
    # Scale CI if present
    for col in ["ci_lower", "ci_upper"]:
        if col in fcst and fcst[col].notna().any():
            fcst[col] = (fcst[col] / combined.loc[base_year]) * 100.0 if base_year in combined.index else fcst[col]

# Dependency ratio (Total−Working)/Working
def dependency_ratio(region_code: str, up_to_year: int):
    # Need both metrics
    t = df[(df["region_code"] == region_code) & (df["metric"] == "total_population")].copy()
    w = df[(df["region_code"] == region_code) & (df["metric"] == "working_age_population")].copy()
    if t.empty or w.empty:
        return None
    t = t[(t["source"] == "historical") & (t["year"] <= up_to_year)].set_index("year")["value"]
    w = w[(w["source"] == "historical") & (w["year"] <= up_to_year)].set_index("year")["value"]
    common = t.index.intersection(w.index)
    if common.empty:
        return None
    dep = (t.loc[common] - w.loc[common]) / w.loc[common]
    return dep

dep_hist = dependency_ratio(sel_code, last_hist_year)

# KPIs
left, mid, right, right2 = st.columns(4)
def fmt(x): 
    return "—" if pd.isna(x) else f"{x:,.0f}"

# Last hist value
last_val = float(hist_s.iloc[-1]) if len(hist_s) else np.nan
left.metric(f"{metric_label} — {last_hist_year}", fmt(last_val))

# 2030 or clip-year forecast value
if len(fcst_s):
    target_year = int(fcst_s.index.max())
    target_val = float(fcst_s.iloc[-1])
    mid.metric(f"{metric_label} — {target_year}", fmt(target_val))

    # CAGR from last_hist_year to target_year
    years = target_year - last_hist_year
    if years > 0 and last_val > 0:
        cagr = (target_val / last_val) ** (1 / years) - 1
        right.metric(f"CAGR {last_hist_year}–{target_year}", f"{cagr*100:.2f}%")
    else:
        right.metric(f"CAGR {last_hist_year}–{target_year}", "—")
else:
    mid.metric(f"{metric_label} — {clip_year}", "—")
    right.metric(f"CAGR {last_hist_year}–{clip_year}", "—")

# Dependency ratio at last_hist_year (Total−Working)/Working
if dep_hist is not None and not dep_hist.empty:
    dr = float(dep_hist.iloc[-1])
    right2.metric("Dependency ratio (hist)", f"{dr*100:.1f}%")
else:
    right2.metric("Dependency ratio (hist)", "—")

# Build chart
title = f"{sel_region} — {metric_label} ({min_year}–{clip_year})"
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=hist_s.index.astype(int), y=hist_s.values,
    mode="lines", name="Historical"
))

# Forecast
if len(fcst_s):
    fig.add_trace(go.Scatter(
        x=fcst_s.index.astype(int), y=fcst_s.values,
        mode="lines", name="Forecast", line=dict(dash="dash")
    ))

    # CI shading if available
    if "ci_lower" in fcst.columns and fcst["ci_lower"].notna().any():
        # Upper bound
        fig.add_trace(go.Scatter(
            x=fcst["year"].astype(int), y=fcst["ci_upper"].values,
            mode="lines", name="CI Upper", hoverinfo="skip", showlegend=False
        ))
        # Lower bound with fill to upper
        fig.add_trace(go.Scatter(
            x=fcst["year"].astype(int), y=fcst["ci_lower"].values,
            mode="lines", name="CI Lower", fill="tonexty", hoverinfo="skip", showlegend=False
        ))

# Vertical line at last historical year
fig.add_vline(x=last_hist_year, line_width=1, line_dash="dot")

fig.update_layout(
    title=title,
    xaxis_title="Year",
    yaxis_title=("Index (base=100)" if index_on else "People"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)

st.plotly_chart(fig, use_container_width=True)

# Optional raw table toggle
with st.expander("Show data table"):
    show_cols = ["year", "value", "source"]
    if "ci_lower" in fcst.columns:
        show_cols += ["ci_lower", "ci_upper"]
    view = pd.concat([hist[show_cols], fcst[show_cols]], axis=0)
    st.dataframe(view.reset_index(drop=True))
