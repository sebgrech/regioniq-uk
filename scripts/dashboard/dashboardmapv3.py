# scripts/dashboard/dashboard_full.py
# Region IQ — Unified ITL1 Dashboard (Population + Incomes, Chart + Choropleth)

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------- Page & Theme -----------------------------
st.set_page_config(page_title="Region IQ — ITL1 (Population + Incomes)", layout="wide")
st.markdown(
    """
<style>
.reportview-container .main .block-container{padding-top:1rem;padding-bottom:2rem;}
[data-testid="stMetricValue"] { font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------- Paths & Config ---------------------------
POP_LONG   = Path("data/forecast/population_ITL1_hist_forecast_long.csv")
INC_LONG   = Path("data/forecast/income_GDHI_ITL1_hist_forecast_long.csv")
GEO_ITL1   = Path("data/geo/ITL1_simplified_clean.geojson")  # same file as before

# Fact tables use E12/W/S/N region codes (e.g., E12000003). Map to ITL1.
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

# Income metric ids
METRIC_PER_HEAD = "gdhi_per_head_gbp"
METRIC_TOTAL_MN = "gdhi_total_mn_gbp"

# ----------------------------- Caching ---------------------------------
@st.cache_resource
def load_geojson(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {path}")
    return json.loads(path.read_text())

@st.cache_resource
def load_long(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data: {path}")
    df = pd.read_csv(path)
    # Schema hygiene
    for c in ["ci_lower", "ci_upper"]:
        if c not in df.columns:
            df[c] = np.nan
    df["year"]  = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for c in ["ci_lower", "ci_upper"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ----------------------------- Helpers ---------------------------------
def fmt_num(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))): return "—"
    try:  return f"{float(v):,.0f}"
    except Exception: return "—"

def fmt_gbp(v, decimals=0):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))): return "—"
    try:  return f"£{float(v):,.{decimals}f}"
    except Exception: return "—"

def fmt_gbp_mn(v, decimals=0):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))): return "—"
    try:  return f"£{float(v):,.{decimals}f}m"
    except Exception: return "—"

def fmt_pct(v, d=2):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))): return "—"
    try:  return f"{float(v):.{d}f}%"
    except Exception: return "—"

def indexify(series: pd.Series, base_year: int | None):
    if base_year is None or base_year not in series.index or pd.isna(series.loc[base_year]) or series.loc[base_year] == 0:
        return series
    return (series / series.loc[base_year]) * 100.0

def detect_featureid_key(gj: dict) -> str:
    keys = list(gj["features"][0]["properties"].keys())
    if "ITL125CD" in keys: return "properties.ITL125CD"
    if "ITL121CD" in keys: return "properties.ITL121CD"
    cands = [k for k in keys if k.startswith("ITL") and k.endswith("CD")]
    return f"properties.{cands[0]}" if cands else "properties.ITL125CD"

def value_at_year(df_long: pd.DataFrame, code: str, metric: str, year: int) -> float | None:
    s = df_long[(df_long["region_code"] == code) & (df_long["metric"] == metric)]
    if s.empty: return None
    ex = s[s["year"] == year]
    if not ex.empty: return float(ex["value"].iloc[0])
    s2 = s[s["year"] <= year].sort_values("year")
    return float(s2["value"].iloc[-1]) if not s2.empty else None

def last_hist_year_for_metric(df_long: pd.DataFrame, code: str, metric: str) -> int | None:
    s = df_long[(df_long["region_code"] == code) & (df_long["metric"] == metric) & (df_long["source"] == "historical")]
    if s.empty: return None
    return int(s["year"].max())

def cagr(v0: float | None, v1: float | None, y0: int | None, y1: int | None) -> float | None:
    if v0 is None or v1 is None or y0 is None or y1 is None or y1 <= y0 or v0 <= 0: return None
    return (v1 / v0) ** (1 / (y1 - y0)) - 1

def dependency_ratio_hist(pop_long: pd.DataFrame, region_code: str, up_to_year: int):
    t = pop_long[(pop_long["region_code"] == region_code) & (pop_long["metric"] == "total_population")]
    w = pop_long[(pop_long["region_code"] == region_code) & (pop_long["metric"] == "working_age_population")]
    if t.empty or w.empty: return None
    t = t[(t["source"] == "historical") & (t["year"] <= up_to_year)].set_index("year")["value"]
    w = w[(w["source"] == "historical") & (w["year"] <= up_to_year)].set_index("year")["value"]
    common = t.index.intersection(w.index)
    if common.empty: return None
    return (t.loc[common] - w.loc[common]) / w.loc[common]

# ----------------------------- Map summaries ----------------------------
def pop_map_summary(pop_long: pd.DataFrame, horizon_year: int) -> pd.DataFrame:
    last_hist = pop_long[pop_long["source"] == "historical"].groupby("region_code")["year"].max().astype(int)
    rows = []
    for code, lh in last_hist.items():
        name  = pop_long.loc[pop_long["region_code"] == code, "region"].iloc[0]
        tot_h = value_at_year(pop_long, code, "total_population", lh)
        tot_y = value_at_year(pop_long, code, "total_population", horizon_year)
        cg    = cagr(tot_h, tot_y, lh, horizon_year)
        rows.append({
            "region_code": code,
            "region": name,
            "itl1_code": E12_TO_ITL1.get(code),
            f"total_{horizon_year}": tot_y,
            "cagr": (cg * 100.0) if cg is not None else None,
            "last_hist_year": lh,
            "total_last_hist": tot_h,
        })
    return pd.DataFrame(rows)

def income_map_summary(inc_long: pd.DataFrame, horizon_year: int, color_mode: str):
    rows = []
    for code in inc_long["region_code"].drop_duplicates().tolist():
        name = inc_long.loc[inc_long["region_code"] == code, "region"].iloc[0]
        lh_ph   = last_hist_year_for_metric(inc_long, code, METRIC_PER_HEAD)
        ph_last = value_at_year(inc_long, code, METRIC_PER_HEAD, lh_ph) if lh_ph else None
        ph_hzn  = value_at_year(inc_long, code, METRIC_PER_HEAD, horizon_year)
        tot_hzn = value_at_year(inc_long, code, METRIC_TOTAL_MN, horizon_year)
        ph_cagr = cagr(ph_last, ph_hzn, lh_ph, horizon_year)
        rows.append({
            "region_code": code,
            "region": name,
            "itl1_code": E12_TO_ITL1.get(code),
            "per_head_hzn": ph_hzn,
            "per_head_cagr_pct": (ph_cagr * 100.0) if ph_cagr is not None else None,
            "total_hzn_mn": tot_hzn,
            "last_hist_year_ph": lh_ph,
            "per_head_last_hist": ph_last,
        })
    dfm = pd.DataFrame(rows)
    if color_mode == "per_head_cagr":
        dfm["z"] = dfm["per_head_cagr_pct"]; cbar = f"Per-head CAGR to {horizon_year} (%)"
    elif color_mode == "per_head_level":
        dfm["z"] = dfm["per_head_hzn"];      cbar = f"Per-head GDHI {horizon_year} (£)"
    else:
        dfm["z"] = dfm["total_hzn_mn"];      cbar = f"Total GDHI {horizon_year} (£m)"
    return dfm, cbar

# ----------------------------- Map renderers ----------------------------
def render_pop_map(pop_long: pd.DataFrame, horizon_year: int, mode: str):
    gj = load_geojson(GEO_ITL1); featureidkey = detect_featureid_key(gj)
    summary = pop_map_summary(pop_long, horizon_year)
    if "itl1_code" not in summary or summary["itl1_code"].isna().any():
        st.error("Missing ITL1 codes for some regions (check E12→ITL1 mapping)."); return

    if mode == "cagr":
        z = summary["cagr"].fillna(0.0); cbar = f"CAGR to {horizon_year} (%)"
    else:
        z = summary[f"total_{horizon_year}"].fillna(0.0); cbar = f"Total population {horizon_year}"

    def hrow(r):
        tot = fmt_num(r.get(f"total_{horizon_year}"))
        cg  = fmt_pct(r.get("cagr"), 2)
        return f"<b>{r.get('region')}</b><br>Total {horizon_year}: {tot}<br>CAGR: {cg}"

    hover = [hrow(row) for row in summary.to_dict(orient="records")]

    fig = go.Figure(go.Choropleth(
        geojson=gj, locations=summary["itl1_code"].astype(str), featureidkey=featureidkey,
        z=z, colorscale="Blues", colorbar_title=cbar,
        marker_line_width=0.5, marker_line_color="rgba(255,255,255,0.7)",
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator",
                    projection_scale=1, center={"lat": 55, "lon": -3})
    fig.update_layout(title=f"ITL1 — Population Map — {cbar}",
                      margin=dict(l=0, r=0, t=50, b=0), height=650)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Map data"):
        cols = ["region", "itl1_code", f"total_{horizon_year}", "cagr"]
        st.dataframe(summary[cols].sort_values("region").reset_index(drop=True))
        st.download_button("Download map summary (CSV)",
                           summary.to_csv(index=False).encode("utf-8"),
                           f"itl1_population_map_{horizon_year}_{mode}.csv", "text/csv")

def render_income_map(inc_long: pd.DataFrame, horizon_year: int, color_mode: str):
    gj = load_geojson(GEO_ITL1); featureidkey = detect_featureid_key(gj)
    summary, cbar = income_map_summary(inc_long, horizon_year, color_mode)
    if "itl1_code" not in summary or summary["itl1_code"].isna().any():
        st.error("Missing ITL1 codes for some regions (check E12→ITL1 mapping)."); return

    def hrow(r):
        ph  = fmt_gbp(r.get("per_head_hzn"))
        tot = fmt_gbp_mn(r.get("total_hzn_mn"))
        cg  = fmt_pct(r.get("per_head_cagr_pct"), 2)
        return f"<b>{r.get('region')}</b><br>Per-head {horizon_year}: {ph}<br>Total {horizon_year}: {tot}<br>Per-head CAGR: {cg}"

    fig = go.Figure(go.Choropleth(
        geojson=gj, locations=summary["itl1_code"].astype(str), featureidkey=featureidkey,
        z=summary["z"].fillna(0.0), colorscale="Blues", colorbar_title=cbar,
        marker_line_width=0.5, marker_line_color="rgba(255,255,255,0.7)",
        hovertext=[hrow(r) for r in summary.to_dict(orient="records")], hoverinfo="text",
    ))
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator",
                    projection_scale=1, center={"lat": 55, "lon": -3})
    fig.update_layout(title=f"ITL1 — Incomes (GDHI) Map — {cbar}",
                      margin=dict(l=0, r=0, t=50, b=0), height=650)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Map data"):
        cols = ["region", "itl1_code", "per_head_hzn", "per_head_cagr_pct", "total_hzn_mn"]
        st.dataframe(summary[cols].sort_values("region").reset_index(drop=True))
        st.download_button("Download income map summary (CSV)",
                           summary.to_csv(index=False).encode("utf-8"),
                           f"itl1_income_map_{horizon_year}_{color_mode}.csv", "text/csv")

# ----------------------------- Data Load --------------------------------
try:
    df_pop = load_long(POP_LONG)
    df_inc = load_long(INC_LONG)
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

max_year = int(max(df_pop["year"].max(), df_inc["year"].max()))

# ----------------------------- Sidebar ----------------------------------
st.sidebar.title("Controls")

dataset_label = st.sidebar.radio("Dataset", ["Population", "Income (GDHI)"], index=1, horizontal=True)
view         = st.sidebar.radio("View", ["Chart", "Map"], index=0, horizontal=True)

has_2050 = max_year >= 2050
horizon_label = st.sidebar.radio("Horizon", ["2030"] + (["Full"] if has_2050 else []), index=0)
clip_year = 2030 if horizon_label == "2030" else max_year

# Dataset-specific controls
if dataset_label == "Population":
    df_cur = df_pop.copy()
    metric_map = {
        "Total population": "total_population",
        "Working-age (16–64)": "working_age_population",
    }
else:
    df_cur = df_inc.copy()
    metric_map = {
        "Per-head GDHI (£)": METRIC_PER_HEAD,
        "Total GDHI (£m)": METRIC_TOTAL_MN,
    }

if view == "Chart":
    regions = (df_cur[["region_code", "region"]]
               .drop_duplicates()
               .sort_values(["region_code"])
               .assign(label=lambda x: x["region"] + " (" + x["region_code"] + ")"))
    region_label = st.sidebar.selectbox("Region (ITL1)", regions["label"].to_list(), index=0)
    sel_code   = regions.loc[regions["label"] == region_label, "region_code"].iloc[0]
    sel_region = regions.loc[regions["label"] == region_label, "region"].iloc[0]

    metric_label = st.sidebar.radio("Metric", list(metric_map.keys()), index=0)
    sel_metric   = metric_map[metric_label]

    index_on = st.sidebar.checkbox("Index to base year = 100", value=False)
else:
    if dataset_label == "Population":
        pop_map_mode_label = st.sidebar.radio("Map value", ["CAGR to horizon (%)", "Total at horizon"], index=0)
        pop_map_mode = "cagr" if pop_map_mode_label.startswith("CAGR") else "total"
    else:
        color_mode_label = st.sidebar.radio(
            "Map value",
            ["Per-head CAGR to horizon (%)", "Per-head level at horizon (£)", "Total GDHI at horizon (£m)"],
            index=0
        )
        color_mode = {
            "Per-head CAGR to horizon (%)": "per_head_cagr",
            "Per-head level at horizon (£)": "per_head_level",
            "Total GDHI at horizon (£m)": "total_level",
        }[color_mode_label]

# ----------------------------- Main Views -------------------------------
if view == "Map":
    if dataset_label == "Population":
        render_pop_map(df_pop, clip_year, pop_map_mode)
    else:
        render_income_map(df_inc, clip_year, color_mode)

else:
    # ---- Chart: history vs forecast for (dataset, region, metric) ----
    series = (df_cur[(df_cur["region_code"] == sel_code) & (df_cur["metric"] == sel_metric)]
              .dropna(subset=["year", "value"])
              .sort_values("year"))
    if series.empty:
        st.warning("No data for selection."); st.stop()

    # KPI blocks
    if dataset_label == "Population":
        last_hist_year = int(series.loc[series["source"] == "historical", "year"].max())
        min_year = int(series["year"].min())
        max_avail_year = int(series["year"].max())
        clip_year = min(clip_year, max_avail_year)

        # Dependency ratio (hist) from population df
        dep_hist = dependency_ratio_hist(df_pop, sel_code, last_hist_year)

        # Base year slider
        base_year = st.sidebar.slider("Base year", min_year, last_hist_year, min(last_hist_year, 2015)) if index_on else None

        # Split
        hist = series[(series["source"] == "historical") & (series["year"] <= clip_year)].copy()
        fcst = series[(series["source"] == "forecast")  & (series["year"] <= clip_year)].copy()
        hist_s = hist.set_index("year")["value"]; fcst_s = fcst.set_index("year")["value"]

        if index_on:
            combined = indexify(pd.concat([hist_s, fcst_s]), base_year)
            hist_s = combined.loc[hist_s.index]; fcst_s = combined.loc[fcst_s.index]
            if "ci_lower" in fcst.columns and base_year in combined.index and not pd.isna(combined.loc[base_year]):
                for col in ["ci_lower", "ci_upper"]:
                    if col in fcst and fcst[col].notna().any():
                        fcst[col] = (fcst[col] / combined.loc[base_year]) * 100.0

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        last_val = float(hist_s.iloc[-1]) if len(hist_s) else np.nan
        k1.metric(f"{metric_label} — {last_hist_year}", fmt_num(last_val))
        if len(fcst_s):
            target_year = int(fcst_s.index.max()); target_val = float(fcst_s.iloc[-1])
            k2.metric(f"{metric_label} — {target_year}", fmt_num(target_val))
            years = target_year - last_hist_year
            cagr_val = (target_val / last_val) ** (1 / years) - 1 if (years > 0 and last_val > 0) else None
            k3.metric(f"CAGR {last_hist_year}–{target_year}", fmt_pct((cagr_val or 0) * 100))
        else:
            k2.metric(f"{metric_label} — {clip_year}", "—"); k3.metric(f"CAGR {last_hist_year}–{clip_year}", "—")
        if dep_hist is not None and not dep_hist.empty:
            k4.metric("Dependency ratio (hist)", fmt_pct(float(dep_hist.iloc[-1]) * 100, 1))
        else:
            k4.metric("Dependency ratio (hist)", "—")

        # Chart
        y_label = "Index (base=100)" if index_on else "People"
        title = f"{sel_region} — {metric_label} ({min_year}–{clip_year})"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_s.index.astype(int), y=hist_s.values, mode="lines", name="Historical"))
        if len(fcst_s):
            fig.add_trace(go.Scatter(x=fcst_s.index.astype(int), y=fcst_s.values, mode="lines",
                                     name="Forecast", line=dict(dash="dash")))
            if "ci_lower" in fcst.columns and fcst["ci_lower"].notna().any():
                fig.add_trace(go.Scatter(x=fcst["year"].astype(int), y=fcst["ci_upper"].values,
                                         mode="lines", name="CI Upper", hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=fcst["year"].astype(int), y=fcst["ci_lower"].values,
                                         mode="lines", name="CI Lower", fill="tonexty",
                                         hoverinfo="skip", showlegend=False))
        fig.add_vline(x=last_hist_year, line_width=1, line_dash="dot")
        fig.update_layout(title=title, xaxis_title="Year", yaxis_title=y_label,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                          margin=dict(l=10, r=10, t=60, b=10), height=520)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show data table"):
            show_cols = ["year", "value", "source", "ci_lower", "ci_upper"]
            view_tbl = pd.concat([hist[show_cols], fcst[show_cols]], axis=0).reset_index(drop=True)
            st.dataframe(view_tbl)
            st.download_button("Download series (CSV)",
                               view_tbl.to_csv(index=False).encode("utf-8"),
                               f"{sel_code}_{sel_metric}_series.csv", "text/csv")

    else:
        # ---- Income KPIs (per-head focus + total at horizon) ----
        last_hist_year_ph = last_hist_year_for_metric(df_inc, sel_code, METRIC_PER_HEAD)
        per_head_last     = value_at_year(df_inc, sel_code, METRIC_PER_HEAD, last_hist_year_ph) if last_hist_year_ph else None
        per_head_target   = value_at_year(df_inc, sel_code, METRIC_PER_HEAD, clip_year)
        per_head_cagr     = cagr(per_head_last, per_head_target, last_hist_year_ph, clip_year)
        total_target_mn   = value_at_year(df_inc, sel_code, METRIC_TOTAL_MN, clip_year)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Per-head GDHI — {last_hist_year_ph or '—'}", fmt_gbp(per_head_last))
        k2.metric(f"Per-head GDHI — {clip_year}", fmt_gbp(per_head_target))
        k3.metric(f"Per-head CAGR {last_hist_year_ph or '—'}–{clip_year}",
                  fmt_pct((per_head_cagr or 0) * 100))
        k4.metric(f"Total GDHI — {clip_year}", fmt_gbp_mn(total_target_mn))

        # Time bounds for selected metric
        series_sel = (df_inc[(df_inc["region_code"] == sel_code) & (df_inc["metric"] == sel_metric)]
                      .dropna(subset=["year", "value"]).sort_values("year"))
        last_hist_year_any = int(series_sel.loc[series_sel["source"] == "historical", "year"].max())
        min_year = int(series_sel["year"].min())
        max_avail_year = int(series_sel["year"].max())
        clip_year = min(clip_year, max_avail_year)

        base_year = st.sidebar.slider("Base year", min_year, last_hist_year_any, min(last_hist_year_any, 2015)) if index_on else None

        hist = series_sel[(series_sel["source"] == "historical") & (series_sel["year"] <= clip_year)].copy()
        fcst = series_sel[(series_sel["source"] == "forecast")  & (series_sel["year"] <= clip_year)].copy()
        hist_s = hist.set_index("year")["value"]; fcst_s = fcst.set_index("year")["value"]

        if index_on:
            combined = indexify(pd.concat([hist_s, fcst_s]), base_year)
            hist_s = combined.loc[hist_s.index]; fcst_s = combined.loc[fcst_s.index]
            if "ci_lower" in fcst.columns and base_year in combined.index and not pd.isna(combined.loc[base_year]):
                for col in ["ci_lower", "ci_upper"]:
                    if col in fcst and fcst[col].notna().any():
                        fcst[col] = (fcst[col] / combined.loc[base_year]) * 100.0

        y_label = "Index (base=100)" if index_on else ("£ per head" if sel_metric == METRIC_PER_HEAD else "£m")
        title = f"{sel_region} — {metric_label} ({min_year}–{clip_year})"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_s.index.astype(int), y=hist_s.values, mode="lines", name="Historical"))
        if len(fcst_s):
            fig.add_trace(go.Scatter(x=fcst_s.index.astype(int), y=fcst_s.values, mode="lines",
                                     name="Forecast", line=dict(dash="dash")))
            if sel_metric == METRIC_PER_HEAD and "ci_lower" in fcst.columns and fcst["ci_lower"].notna().any():
                fig.add_trace(go.Scatter(x=fcst["year"].astype(int), y=fcst["ci_upper"].values,
                                         mode="lines", name="CI Upper", hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=fcst["year"].astype(int), y=fcst["ci_lower"].values,
                                         mode="lines", name="CI Lower", fill="tonexty",
                                         hoverinfo="skip", showlegend=False))
        fig.add_vline(x=last_hist_year_any, line_width=1, line_dash="dot")
        fig.update_layout(title=title, xaxis_title="Year", yaxis_title=y_label,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                          margin=dict(l=10, r=10, t=60, b=10), height=520)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show data table"):
            show_cols = ["year", "value", "source", "ci_lower", "ci_upper"]
            view_tbl = pd.concat([hist[show_cols], fcst[show_cols]], axis=0).reset_index(drop=True)
            st.dataframe(view_tbl)
            st.download_button("Download series (CSV)",
                               view_tbl.to_csv(index=False).encode("utf-8"),
                               f"{sel_code}_{sel_metric}_series.csv", "text/csv")

# ----------------------------- Footnote ---------------------------------
if dataset_label == "Income (GDHI)":
    st.caption("Coherent system: **Total GDHI (£m)** = **Per-head (£)** × **Population**. "
               "Per-head forecasts may include confidence bands where available.")
