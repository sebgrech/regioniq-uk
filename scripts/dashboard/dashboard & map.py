# scripts/dashboard/dashboard.py
# Region IQ — ITL1 Population Dashboard (Chart + Choropleth)
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------- Page & Theme -----------------------------
st.set_page_config(page_title="Region IQ — ITL1 Population", layout="wide")
st.markdown(
    """
<style>
/* Subtle polish */
.reportview-container .main .block-container{padding-top:1rem;padding-bottom:2rem;}
[data-testid="stMetricValue"] { font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)
stre
# ----------------------------- Paths & Config ---------------------------
DATA_LONG = Path("data/forecast/population_ITL1_hist_forecast_long.csv")
GEO_ITL1_PATH = Path("data/geo/uk_itl1_2025_bgc.geojson")  # BGC file

# Your fact table uses E12/W/S/N region codes (e.g., E12000003). We map to ITL1.
E12_TO_ITL1 = {
    "E12000001": "TLC",  # North East (England)
    "E12000002": "TLD",  # North West (England)
    "E12000003": "TLE",  # Yorkshire and The Humber
    "E12000004": "TLF",  # East Midlands (England)
    "E12000005": "TLG",  # West Midlands (England)
    "E12000006": "TLH",  # East of England
    "E12000007": "TLI",  # London
    "E12000008": "TLJ",  # South East (England)
    "E12000009": "TLK",  # South West (England)
    "W92000004": "TLL",  # Wales
    "S92000003": "TLM",  # Scotland
    "N92000002": "TLN",  # Northern Ireland
}

# ----------------------------- Caching ---------------------------------
@st.cache_resource
def load_geojson(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {path}")
    # Large file → use read_text (fast enough) + json.loads
    return json.loads(path.read_text())

@st.cache_resource
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data: {path}. Run the forecast script first.")
    df = pd.read_csv(path)

    # Ensure schema robustness
    for c in ["ci_lower", "ci_upper"]:
        if c not in df.columns:
            df[c] = np.nan

    # Types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for c in ["ci_lower", "ci_upper"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ----------------------------- Helpers ---------------------------------
def fmt_num(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    try:
        return f"{float(v):,.0f}"
    except Exception:
        return "—"

def fmt_pct(v, decimals=2):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    try:
        return f"{float(v):.{decimals}f}%"
    except Exception:
        return "—"

def indexify(series: pd.Series, base_year: int | None):
    if base_year is None or base_year not in series.index or pd.isna(series.loc[base_year]) or series.loc[base_year] == 0:
        return series
    return (series / series.loc[base_year]) * 100.0

def dependency_ratio_hist(df_long: pd.DataFrame, region_code: str, up_to_year: int):
    t = df_long[(df_long["region_code"] == region_code) & (df_long["metric"] == "total_population")]
    w = df_long[(df_long["region_code"] == region_code) & (df_long["metric"] == "working_age_population")]
    if t.empty or w.empty:
        return None
    t = t[(t["source"] == "historical") & (t["year"] <= up_to_year)].set_index("year")["value"]
    w = w[(w["source"] == "historical") & (w["year"] <= up_to_year)].set_index("year")["value"]
    common = t.index.intersection(w.index)
    if common.empty:
        return None
    dep = (t.loc[common] - w.loc[common]) / w.loc[common]
    return dep

def detect_featureid_key(gj: dict) -> str:
    # Your checker showed: keys include ITL125CD / ITL125NM
    keys = list(gj["features"][0]["properties"].keys())
    # Prefer ITL125CD, else ITL121CD, else the first ITL*CD key
    if "ITL125CD" in keys:
        return "properties.ITL125CD"
    if "ITL121CD" in keys:
        return "properties.ITL121CD"
    candidates = [k for k in keys if k.startswith("ITL") and k.endswith("CD")]
    return f"properties.{candidates[0]}" if candidates else "properties.ITL125CD"

def value_at_year(df_long: pd.DataFrame, code: str, metric: str, year: int) -> float | None:
    s = df_long[(df_long["region_code"] == code) & (df_long["metric"] == metric)]
    if s.empty:
        return None
    exact = s[s["year"] == year]
    if not exact.empty:
        return float(exact["value"].iloc[0])
    s2 = s[s["year"] <= year].sort_values("year")
    return float(s2["value"].iloc[-1]) if not s2.empty else None

def cagr(v0: float | None, v1: float | None, y0: int, y1: int) -> float | None:
    if v0 is None or v1 is None or y1 <= y0 or v0 <= 0:
        return None
    years = y1 - y0
    return (v1 / v0) ** (1 / years) - 1

def build_map_summary(df_long: pd.DataFrame, horizon_year: int) -> pd.DataFrame:
    last_hist = (
        df_long[df_long["source"] == "historical"]
        .groupby("region_code")["year"].max()
        .astype(int)
    )
    rows = []
    for code, lh in last_hist.items():
        name = df_long.loc[df_long["region_code"] == code, "region"].iloc[0]
        tot_h = value_at_year(df_long, code, "total_population", lh)
        tot_y = value_at_year(df_long, code, "total_population", horizon_year)
        wok_y = value_at_year(df_long, code, "working_age_population", horizon_year)
        cg = cagr(tot_h, tot_y, lh, horizon_year)
        rows.append(
            {
                "region_code": code,
                "region": name,
                "itl1_code": E12_TO_ITL1.get(code),
                f"total_{horizon_year}": tot_y,
                f"working_{horizon_year}": wok_y,
                "cagr": (cg * 100.0) if cg is not None else None,  # %
                "last_hist_year": lh,
                "total_last_hist": tot_h,
            }
        )
    return pd.DataFrame(rows)

def render_itl1_map(df_long: pd.DataFrame, horizon_year: int):
    gj = load_geojson(GEO_ITL1_PATH)
    featureidkey = detect_featureid_key(gj)

    summary = build_map_summary(df_long, horizon_year)
    if "itl1_code" not in summary or summary["itl1_code"].isna().any():
        st.error("Missing ITL1 codes for some regions (check E12→ITL1 mapping).")
        return

    # Color by CAGR (%)
    z = summary["cagr"].fillna(0.0)

    def hover_row(r):
        tot = fmt_num(r.get(f"total_{horizon_year}"))
        wok = fmt_num(r.get(f"working_{horizon_year}"))
        cg = fmt_pct(r.get("cagr"), decimals=2)
        return f"<b>{r.get('region')}</b><br>Total {horizon_year}: {tot}<br>Working {horizon_year}: {wok}<br>CAGR to {horizon_year}: {cg}"

    hover = [hover_row(row) for row in summary.to_dict(orient="records")]

    fig = go.Figure(
        go.Choropleth(
            geojson=gj,
            locations=summary["itl1_code"].astype(str),
            featureidkey=featureidkey,
            z=z,
            colorscale="Blues",
            colorbar_title=f"CAGR to {horizon_year} (%)",
            marker_line_width=0.5,
            marker_line_color="rgba(255,255,255,0.7)",
            hovertext=hover,
            hoverinfo="text",
        )
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        title=f"ITL1 — CAGR to {horizon_year} (Total population)",
        margin=dict(l=0, r=0, t=50, b=0),
        height=650,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Map data"):
        st.dataframe(
            summary[["region", "itl1_code", f"total_{horizon_year}", f"working_{horizon_year}", "cagr"]]
            .sort_values("region")
            .reset_index(drop=True)
        )
        csv = summary.to_csv(index=False).encode("utf-8")
        st.download_button("Download map summary (CSV)", csv, f"itl1_map_summary_{horizon_year}.csv", "text/csv")

# ----------------------------- Data Load --------------------------------
try:
    df = load_data(DATA_LONG)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

max_year_in_data = int(df["year"].max())

# ----------------------------- Sidebar ----------------------------------
st.sidebar.title("Controls")

# View toggle first (so we can show/hide controls)
view = st.sidebar.radio("View", ["Chart", "Map"], index=0, horizontal=True)

# Horizon
has_2050 = max_year_in_data >= 2050
horizon_label = st.sidebar.radio("Horizon", ["2030"] + (["Full"] if has_2050 else []), index=0)
clip_year = 2030 if horizon_label == "2030" else max_year_in_data

# Chart controls (only relevant in Chart view)
metric_map = {
    "Total population": "total_population",
    "Working-age (16–64)": "working_age_population",
}

if view == "Chart":
    # Region select
    regions = (
        df[["region_code", "region"]]
        .drop_duplicates()
        .sort_values(["region_code"])
        .assign(label=lambda x: x["region"] + " (" + x["region_code"] + ")")
    )
    region_label = st.sidebar.selectbox("Region (ITL1)", regions["label"].to_list(), index=0)
    sel_code = regions.loc[regions["label"] == region_label, "region_code"].iloc[0]
    sel_region = regions.loc[regions["label"] == region_label, "region"].iloc[0]

    # Metric select
    metric_label = st.sidebar.radio("Metric", list(metric_map.keys()), index=0)
    sel_metric = metric_map[metric_label]

    # Indexing
    index_on = st.sidebar.checkbox("Index to base year = 100", value=False)

# ----------------------------- Main Views --------------------------------
if view == "Map":
    # Choropleth view
    render_itl1_map(df, clip_year)

else:
    # -------- Chart view: history vs forecast for (region, metric) --------
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

    # Base year slider (for index)
    if index_on:
        base_year = st.sidebar.slider("Base year", min_year, last_hist_year, min(last_hist_year, 2015))
    else:
        base_year = None

    # Split hist & forecast
    hist = series[(series["source"] == "historical") & (series["year"] <= clip_year)].copy()
    fcst = series[(series["source"] == "forecast") & (series["year"] <= clip_year)].copy()

    hist_s = hist.set_index("year")["value"]
    fcst_s = fcst.set_index("year")["value"]

    # Optional index scaling
    if index_on:
        combined = pd.concat([hist_s, fcst_s])
        combined = indexify(combined, base_year)
        hist_s = combined.loc[hist_s.index]
        fcst_s = combined.loc[fcst_s.index]
        # Scale CI if present
        if "ci_lower" in fcst.columns and base_year in combined.index and not pd.isna(combined.loc[base_year]):
            for col in ["ci_lower", "ci_upper"]:
                if col in fcst and fcst[col].notna().any():
                    fcst[col] = (fcst[col] / combined.loc[base_year]) * 100.0

    # KPIs
    left, mid, right, right2 = st.columns(4)
    last_val = float(hist_s.iloc[-1]) if len(hist_s) else np.nan
    left.metric(f"{metric_label} — {last_hist_year}", fmt_num(last_val))

    if len(fcst_s):
        target_year = int(fcst_s.index.max())
        target_val = float(fcst_s.iloc[-1])
        mid.metric(f"{metric_label} — {target_year}", fmt_num(target_val))

        years = target_year - last_hist_year
        if years > 0 and last_val > 0:
            cagr_val = (target_val / last_val) ** (1 / years) - 1
            right.metric(f"CAGR {last_hist_year}–{target_year}", fmt_pct(cagr_val * 100))
        else:
            right.metric(f"CAGR {last_hist_year}–{target_year}", "—")
    else:
        mid.metric(f"{metric_label} — {clip_year}", "—")
        right.metric(f"CAGR {last_hist_year}–{clip_year}", "—")

    dep_hist = dependency_ratio_hist(df, sel_code, last_hist_year)
    if dep_hist is not None and not dep_hist.empty:
        right2.metric("Dependency ratio (hist)", fmt_pct(float(dep_hist.iloc[-1]) * 100, 1))
    else:
        right2.metric("Dependency ratio (hist)", "—")

    # Chart
    title = f"{sel_region} — {metric_label} ({min_year}–{clip_year})"
    fig = go.Figure()

    # Historical
    fig.add_trace(
        go.Scatter(
            x=hist_s.index.astype(int),
            y=hist_s.values,
            mode="lines",
            name="Historical",
        )
    )

    # Forecast
    if len(fcst_s):
        fig.add_trace(
            go.Scatter(
                x=fcst_s.index.astype(int),
                y=fcst_s.values,
                mode="lines",
                name="Forecast",
                line=dict(dash="dash"),
            )
        )
        # CI shading if available
        if "ci_lower" in fcst.columns and fcst["ci_lower"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=fcst["year"].astype(int),
                    y=fcst["ci_upper"].values,
                    mode="lines",
                    name="CI Upper",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fcst["year"].astype(int),
                    y=fcst["ci_lower"].values,
                    mode="lines",
                    name="CI Lower",
                    fill="tonexty",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Split marker at boundary of hist/forecast
    fig.add_vline(x=last_hist_year, line_width=1, line_dash="dot")

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=("Index (base=100)" if index_on else "People"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data table + download
    with st.expander("Show data table"):
        show_cols = ["year", "value", "source", "ci_lower", "ci_upper"]
        view_tbl = pd.concat([hist[show_cols], fcst[show_cols]], axis=0).reset_index(drop=True)
        st.dataframe(view_tbl)
        csv = view_tbl.to_csv(index=False).encode("utf-8")
        st.download_button("Download series (CSV)", csv, f"{sel_code}_{sel_metric}_series.csv", "text/csv")
