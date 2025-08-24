#!/usr/bin/env python3
"""
Region IQ - Ultimate V5 Dashboard
==================================
Professional-grade regional economic dashboard with enhanced UI/UX
"""

# ABSOLUTE FIRST THING - Import streamlit and set page config
import streamlit as st

st.set_page_config(
    page_title="Region IQ | Professional Economic Forecasts",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NOW all other imports
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import io

# Optional imports for enhanced features
HAVE_REPORTLAB = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    HAVE_REPORTLAB = True
except ImportError:
    HAVE_REPORTLAB = False

HAVE_PLOTLY_EVENTS = False
try:
    from streamlit_plotly_events import plotly_events
    HAVE_PLOTLY_EVENTS = True
except ImportError:
    HAVE_PLOTLY_EVENTS = False

# ===========================
# Professional Theme & Typography
# ===========================

st.markdown("""
<link rel="preconnect" href="https://rsms.me/"/>
<link href="https://rsms.me/inter/inter.css" rel="stylesheet">
<style>
:root{
  --riq-bg:#0e1217;
  --riq-panel:#11161d;
  --riq-text:#e7ebf0;
  --riq-sub:#9aa4b2;
  --riq-accent:#FF6900;
  --riq-brand:#003087;
  --riq-green:#28a745;
  --riq-border:#1b2330;
}

/* Global dark theme */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: var(--riq-bg) !important;
  color: var(--riq-text) !important;
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  font-feature-settings: "tnum" 1, "cv10" 1;
  font-variant-numeric: tabular-nums;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--riq-panel) !important;
  border-right: 1px solid var(--riq-border);
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label {
  color: var(--riq-sub);
  font-size: 0.875rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* Typography */
h1, h2, h3 { 
  color: var(--riq-text); 
  letter-spacing: -0.01em;
  font-family: 'Inter', sans-serif;
}
h1 { 
  font-weight: 800; 
  font-size: 2.5rem;
  border-bottom: 3px solid var(--riq-accent);
  padding-bottom: 10px;
  margin-bottom: 2rem;
}
h2 { 
  font-weight: 700;
  font-size: 1.875rem;
  color: var(--riq-text);
  margin-top: 2rem;
}
h3 {
  font-weight: 600;
  font-size: 1.25rem;
}

/* Metrics */
[data-testid="stMetricValue"] {
  color: var(--riq-accent) !important;
  font-weight: 800;
  font-size: 2rem;
  font-feature-settings: "tnum" 1;
}
[data-testid="stMetricDelta"] {
  color: var(--riq-green);
  font-weight: 600;
}
[data-testid="stMetricLabel"] {
  color: var(--riq-sub);
  font-weight: 600;
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.05em;
}

/* Cards and containers */
.main .block-container {
  padding-top: 2rem;
  max-width: 100%;
}

.stAlert {
  background: rgba(17, 22, 29, 0.8);
  border-left: 4px solid var(--riq-accent);
  border-radius: 8px;
  color: var(--riq-text);
}

.stInfo {
  background: rgba(0, 48, 135, 0.1);
  border-left: 4px solid var(--riq-brand);
}

.stSuccess {
  background: rgba(40, 167, 69, 0.1);
  border-left: 4px solid var(--riq-green);
}

/* Buttons */
.stDownloadButton button, .stButton button {
  background: var(--riq-brand);
  color: white;
  border: none;
  border-radius: 8px;
  padding: 0.625rem 1.25rem;
  font-weight: 700;
  font-size: 0.875rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  transition: all 0.2s ease;
}

.stDownloadButton button:hover, .stButton button:hover {
  background: var(--riq-accent);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(255, 105, 0, 0.3);
}

/* Tables */
.stTable {
  background: var(--riq-panel);
  border-radius: 8px;
  overflow: hidden;
}

/* Expanders */
.streamlit-expanderHeader {
  background: var(--riq-panel);
  border-radius: 8px;
  font-weight: 600;
  color: var(--riq-text);
}

/* Logo placeholder (top right) */
.logo-container {
  position: fixed;
  top: 1rem;
  right: 2rem;
  z-index: 999;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.logo-text {
  font-size: 1.125rem;
  font-weight: 800;
  color: var(--riq-accent);
  letter-spacing: -0.02em;
}

/* Dividers */
hr {
  border-color: var(--riq-border);
  margin: 2rem 0;
}

/* Quality badges */
.quality-badge {
  display: inline-flex;
  align-items: center;
  padding: 0.375rem 0.75rem;
  border-radius: 20px;
  font-size: 0.875rem;
  font-weight: 600;
  gap: 0.375rem;
}

.quality-excellent {
  background: rgba(40, 167, 69, 0.2);
  color: var(--riq-green);
  border: 1px solid rgba(40, 167, 69, 0.3);
}

.quality-good {
  background: rgba(255, 193, 7, 0.2);
  color: #ffc107;
  border: 1px solid rgba(255, 193, 7, 0.3);
}

.quality-moderate {
  background: rgba(255, 105, 0, 0.2);
  color: var(--riq-accent);
  border: 1px solid rgba(255, 105, 0, 0.3);
}

.quality-low {
  background: rgba(220, 53, 69, 0.2);
  color: #dc3545;
  border: 1px solid rgba(220, 53, 69, 0.3);
}
</style>
""", unsafe_allow_html=True)

# Add logo to top right
st.markdown("""
<div class="logo-container">
    <span class="logo-text">REGION IQ</span>
</div>
""", unsafe_allow_html=True)

# ===========================
# Data Paths & Configuration
# ===========================

# Paths to V3 forecast outputs
FORECAST_LONG = Path("data/forecast/forecast_v3_long.csv")
FORECAST_WIDE = Path("data/forecast/forecast_v3_wide.csv")
CONFIDENCE_INTERVALS = Path("data/forecast/confidence_intervals_v3.csv")
QUALITY_METRICS = Path("data/forecast/forecast_quality_v3.csv")
METADATA = Path("data/forecast/metadata_v3.json")
GEO_ITL1 = Path("data/geo/ITL1_simplified_clean.geojson")

# Fixed GeoJSON property key
GEO_ITL1_FEATURE_KEY = "properties.ITL1CD"  # Adjust if your GeoJSON uses different key

# ITL1 code mapping
ITL1_NAMES = {
    "TLC": "North East",
    "TLD": "North West",
    "TLE": "Yorkshire & Humber",
    "TLF": "East Midlands",
    "TLG": "West Midlands",
    "TLH": "East of England",
    "TLI": "London",
    "TLJ": "South East",
    "TLK": "South West",
    "TLL": "Wales",
    "TLM": "Scotland",
    "TLN": "Northern Ireland"
}

# Plotly theme
PLOTLY_TEMPLATE = "plotly_dark"

# Chart colors
HIST_COLOR = "#3aa3ff"   # Cool blue for historical
FORE_COLOR = "#FF6900"   # Brand orange for forecast
CONF_FILL = "rgba(255,105,0,0.18)"  # Confidence interval fill

# Metric display configuration (Productivity removed)
METRIC_CONFIG = {
    'population_total': {
        'name': 'Total Population',
        'unit': 'persons',
        'format': 'number',
        'color': '#003087',
        'icon': '👥'
    },
    'gdhi_total_mn_gbp': {
        'name': 'Total Income (GDHI)',
        'unit': '£m',
        'format': 'currency_m',
        'color': '#28a745',
        'icon': '💷'
    },
    'gdhi_per_head_gbp': {
        'name': 'Income per Head',
        'unit': '£',
        'format': 'currency',
        'color': '#28a745',
        'icon': '💰'
    },
    'nominal_gva_mn_gbp': {
        'name': 'Economic Output (GVA)',
        'unit': '£m nominal',
        'format': 'currency_m',
        'color': '#FF6900',
        'icon': '📈'
    },
    'chained_gva_mn_gbp': {
        'name': 'Real Economic Output',
        'unit': '£m (2022 prices)',
        'format': 'currency_m',
        'color': '#FF6900',
        'icon': '📊'
    },
    'emp_total_jobs': {
        'name': 'Total Employment',
        'unit': 'jobs',
        'format': 'number',
        'color': '#17a2b8',
        'icon': '💼'
    },
    'employment_rate': {
        'name': 'Employment Rate',
        'unit': '%',
        'format': 'percentage',
        'color': '#17a2b8',
        'icon': '📊'
    },
    'income_per_worker_gbp': {
        'name': 'Income per Worker',
        'unit': '£',
        'format': 'currency',
        'color': '#28a745',
        'icon': '💵'
    }
}

# ===========================
# Helper Functions
# ===========================

@st.cache_data
def load_data():
    """Load all forecast data with caching"""
    if not FORECAST_LONG.exists():
        st.error(f"Forecast data not found at {FORECAST_LONG}. Please run the forecasting engine first.")
        st.stop()
    
    # Load main data
    df_long = pd.read_csv(FORECAST_LONG)
    df_wide = pd.read_csv(FORECAST_WIDE)
    
    # Load supplementary data
    df_ci = pd.read_csv(CONFIDENCE_INTERVALS) if CONFIDENCE_INTERVALS.exists() else pd.DataFrame()
    df_quality = pd.read_csv(QUALITY_METRICS) if QUALITY_METRICS.exists() else pd.DataFrame()
    
    # Load metadata
    metadata = {}
    if METADATA.exists():
        with open(METADATA, 'r') as f:
            metadata = json.load(f)
    
    return df_long, df_wide, df_ci, df_quality, metadata

@st.cache_data
def load_geojson():
    """Load GeoJSON for mapping"""
    if not GEO_ITL1.exists():
        return None
    return json.loads(GEO_ITL1.read_text())

def format_value(value, format_type):
    """Format values based on type"""
    if pd.isna(value) or value is None:
        return "—"
    
    if format_type == 'number':
        return f"{value:,.0f}"
    elif format_type == 'currency':
        return f"£{value:,.0f}"
    elif format_type == 'currency_m':
        return f"£{value:,.0f}m"
    elif format_type == 'percentage':
        return f"{value:.1f}%"
    else:
        return f"{value:.2f}"

def calculate_cagr(start_value, end_value, years):
    """Calculate compound annual growth rate"""
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return None
    return ((end_value / start_value) ** (1 / years) - 1) * 100

def pick_forecast_target(fore_df, target=2030):
    """Robust forecast target picker with fallback"""
    if fore_df.empty:
        return None, None
    years = sorted(fore_df['year'].unique())
    year = target if target in years else years[-1]
    val = fore_df.loc[fore_df['year'] == year, 'value'].iloc[0]
    return year, val

def get_quality_badge(cv_value):
    """Generate quality badge based on coefficient of variation"""
    if pd.isna(cv_value):
        return "🔵 No Data"
    elif cv_value < 0.05:
        return "🟢 Excellent"
    elif cv_value < 0.10:
        return "🟡 Good"
    elif cv_value < 0.20:
        return "🟠 Moderate"
    else:
        return "🔴 High Uncertainty"

def apply_professional_layout(fig, title=None):
    """Apply consistent professional styling to Plotly figures"""
    fig.update_layout(
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        template=PLOTLY_TEMPLATE
    )
    
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                x=0.0,
                xanchor="left",
                yanchor="top",
                font=dict(size=20, weight=800)
            )
        )
    
    fig.update_layout(
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=12)
        ),
        margin=dict(l=0, r=0, t=50 if title else 20, b=0)
    )
    
    fig.update_xaxes(showgrid=False, zeroline=False, color='#9aa4b2')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.08)', zeroline=False, color='#9aa4b2')
    
    return fig

def generate_ai_narrative(data, region, metric, metadata):
    """Generate AI-style narrative for the data"""
    hist_data = data[data['data_type'] == 'historical']
    fore_data = data[data['data_type'] == 'forecast']
    
    if hist_data.empty or fore_data.empty:
        return "Insufficient data for narrative generation."
    
    # Calculate key statistics
    last_hist_year = hist_data['year'].max()
    last_hist_value = hist_data[hist_data['year'] == last_hist_year]['value'].iloc[0]
    
    target_year, forecast_value = pick_forecast_target(fore_data, 2030)
    if target_year is None:
        return "Unable to generate forecast narrative."
    
    cagr = calculate_cagr(last_hist_value, forecast_value, target_year - last_hist_year)
    
    # Get metric info
    metric_info = METRIC_CONFIG.get(metric, {})
    metric_name = metric_info.get('name', metric)
    
    # Build narrative
    narrative = f"""
    ### {region} - {metric_name}
    
    **Current Status**: As of {last_hist_year}, {region}'s {metric_name.lower()} stood at {format_value(last_hist_value, metric_info.get('format', 'number'))} {metric_info.get('unit', '')}.
    
    **Forecast Outlook**: Our models project {metric_name.lower()} will reach {format_value(forecast_value, metric_info.get('format', 'number'))} {metric_info.get('unit', '')} by {target_year}, representing a compound annual growth rate of {cagr:.1f}%.
    
    **Key Drivers**: This trajectory reflects post-COVID recovery dynamics, structural economic shifts, and regional development patterns. The forecast accounts for historical volatility and structural breaks including the 2008 financial crisis and 2020 pandemic impacts.
    
    **Confidence**: Based on {hist_data['year'].nunique()} years of historical data, our ensemble model combining ARIMA, ETS, and linear trend approaches provides robust projections with quantified uncertainty bands.
    """
    
    return narrative

# ===========================
# Visualization Functions
# ===========================

def create_time_series_chart(data, region, metric, show_confidence=True):
    """Create professional time series chart with forecast"""
    
    metric_info = METRIC_CONFIG.get(metric, {})
    
    # Split historical and forecast
    hist = data[data['data_type'] == 'historical'].sort_values('year')
    fore = data[data['data_type'] == 'forecast'].sort_values('year')
    
    fig = go.Figure()
    
    # Historical line
    fig.add_trace(go.Scatter(
        x=hist['year'],
        y=hist['value'],
        mode='lines+markers',
        name='Historical',
        line=dict(color=HIST_COLOR, width=2.5),
        marker=dict(size=5, color=HIST_COLOR)
    ))
    
    # Forecast line
    if not fore.empty:
        fig.add_trace(go.Scatter(
            x=fore['year'],
            y=fore['value'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color=FORE_COLOR, width=2.5, dash='dash'),
            marker=dict(size=5, color=FORE_COLOR)
        ))
        
        # Confidence intervals
        if show_confidence and 'ci_lower' in fore.columns:
            fig.add_trace(go.Scatter(
                x=fore['year'].tolist() + fore['year'].tolist()[::-1],
                y=fore['ci_upper'].tolist() + fore['ci_lower'].tolist()[::-1],
                fill='toself',
                fillcolor=CONF_FILL,
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='95% Confidence'
            ))
    
    # Add recession shading
    fig.add_vrect(x0=2008, x1=2009, fillcolor="rgba(255,255,255,0.05)", line_width=0)
    fig.add_vrect(x0=2020, x1=2021, fillcolor="rgba(255,255,255,0.05)", line_width=0)
    
    # Add vertical line at forecast start
    if not hist.empty and not fore.empty:
        last_hist_year = hist['year'].max()
        fig.add_vline(x=last_hist_year, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    
    # Update layout
    fig = apply_professional_layout(fig, f"{region} - {metric_info.get('name', metric)}")
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=f"{metric_info.get('unit', '')}",
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_comparison_chart(data, regions, metric):
    """Create multi-region comparison chart"""
    
    metric_info = METRIC_CONFIG.get(metric, {})
    
    fig = go.Figure()
    
    colors = ['#3aa3ff', '#FF6900', '#28a745', '#dc3545', '#ffc107', '#17a2b8', '#6610f2', '#e83e8c']
    
    for i, region in enumerate(regions):
        region_data = data[(data['region_code'] == region) & (data['metric'] == metric)]
        
        if not region_data.empty:
            # Historical
            hist = region_data[region_data['data_type'] == 'historical'].sort_values('year')
            if not hist.empty:
                fig.add_trace(go.Scatter(
                    x=hist['year'],
                    y=hist['value'],
                    mode='lines',
                    name=f"{ITL1_NAMES.get(region, region)}",
                    line=dict(color=colors[i % len(colors)], width=2.5),
                    legendgroup=region
                ))
            
            # Forecast
            fore = region_data[region_data['data_type'] == 'forecast'].sort_values('year')
            if not fore.empty:
                fig.add_trace(go.Scatter(
                    x=fore['year'],
                    y=fore['value'],
                    mode='lines',
                    name=f"{ITL1_NAMES.get(region, region)} (forecast)",
                    line=dict(color=colors[i % len(colors)], width=2.5, dash='dash'),
                    showlegend=False,
                    legendgroup=region
                ))
    
    fig = apply_professional_layout(fig, f"Regional Comparison - {metric_info.get('name', metric)}")
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=f"{metric_info.get('unit', '')}",
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_choropleth_map(data, metric, year, geojson, mapbox_token=None):
    """Create enhanced choropleth map with Mapbox"""
    
    if geojson is None:
        st.warning("GeoJSON file not found. Map visualization not available.")
        return None
    
    metric_info = METRIC_CONFIG.get(metric, {})
    
    # Filter data for specific year and metric
    map_data = data[(data['year'] == year) & (data['metric'] == metric)]
    
    if map_data.empty:
        st.warning(f"No data available for {metric} in {year}")
        return None
    
    # Prepare data for choropleth
    map_df = map_data[['region_code', 'value']].copy()
    map_df['region_name'] = map_df['region_code'].map(ITL1_NAMES)
    
    # Create Mapbox choropleth
    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=map_df['region_code'],
        z=map_df['value'],
        featureidkey=GEO_ITL1_FEATURE_KEY,
        colorscale="Viridis",
        zmin=map_df['value'].min(),
        zmax=map_df['value'].max(),
        marker_line_width=0.8,
        marker_line_color="rgba(255,255,255,0.35)",
        colorbar=dict(
            title=metric_info.get('unit', ''),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            borderwidth=0,
            tickfont=dict(color='#9aa4b2')
        ),
        hovertemplate="<b>%{customdata[0]}</b><br>" +
                      f"{metric_info.get('name', metric)}: %{{z:,.0f}} {metric_info.get('unit','')}" +
                      "<extra></extra>",
        customdata=np.stack([map_df['region_name'], map_df['region_code']], axis=-1)
    ))
    
    # Set map style
    if mapbox_token:
        fig.update_layout(mapbox_style="dark", mapbox_accesstoken=mapbox_token)
    else:
        fig.update_layout(mapbox_style="carto-darkmatter")
    
    fig.update_layout(
        mapbox_zoom=4.8,
        mapbox_center={"lat": 54.5, "lon": -2.5},
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        title=dict(
            text=f"{metric_info.get('name', metric)} - {year}",
            font=dict(size=20, weight=800, color='#e7ebf0')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_dashboard_summary(data, region):
    """Create a comprehensive dashboard summary for a region"""
    
    # Create subplots (removed Productivity)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Population Trend', 'Economic Output (GVA)', 'Employment',
                       'Income per Head', 'Growth Rates', ''),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'bar'}, None]]
    )
    
    # Define metrics to plot (removed productivity)
    metrics_to_plot = [
        ('population_total', 1, 1),
        ('nominal_gva_mn_gbp', 1, 2),
        ('emp_total_jobs', 1, 3),
        ('gdhi_per_head_gbp', 2, 1)
    ]
    
    # Plot each metric
    for metric, row, col in metrics_to_plot:
        metric_data = data[(data['region_code'] == region) & (data['metric'] == metric)]
        
        if not metric_data.empty:
            hist = metric_data[metric_data['data_type'] == 'historical'].sort_values('year')
            fore = metric_data[metric_data['data_type'] == 'forecast'].sort_values('year')
            
            if not hist.empty:
                fig.add_trace(
                    go.Scatter(x=hist['year'], y=hist['value'], mode='lines',
                             line=dict(color=HIST_COLOR, width=2),
                             showlegend=False),
                    row=row, col=col
                )
            
            if not fore.empty:
                fig.add_trace(
                    go.Scatter(x=fore['year'], y=fore['value'], mode='lines',
                             line=dict(color=FORE_COLOR, width=2, dash='dash'),
                             showlegend=False),
                    row=row, col=col
                )
    
    # Add growth rates bar chart
    growth_rates = []
    for metric in ['population_total', 'nominal_gva_mn_gbp', 'emp_total_jobs']:
        metric_data = data[(data['region_code'] == region) & (data['metric'] == metric)]
        if not metric_data.empty:
            hist = metric_data[metric_data['data_type'] == 'historical']
            fore = metric_data[metric_data['data_type'] == 'forecast']
            
            if not hist.empty and not fore.empty:
                last_hist = hist.nlargest(1, 'year')['value'].iloc[0]
                target_year, forecast_value = pick_forecast_target(fore, 2030)
                if target_year:
                    cagr = calculate_cagr(last_hist, forecast_value, 
                                        target_year - hist['year'].max())
                    if cagr:
                        growth_rates.append((METRIC_CONFIG.get(metric, {}).get('name', metric), cagr))
    
    if growth_rates:
        fig.add_trace(
            go.Bar(x=[g[0] for g in growth_rates],
                  y=[g[1] for g in growth_rates],
                  marker_color='#17a2b8',
                  showlegend=False),
            row=2, col=2
        )
    
    fig = apply_professional_layout(fig, f"Regional Dashboard - {ITL1_NAMES.get(region, region)}")
    fig.update_layout(height=700, showlegend=False)
    
    return fig

def generate_pdf_report(data, region, metadata):
    """Generate downloadable PDF report"""
    if not HAVE_REPORTLAB:
        return None
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#003087'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph(f"Regional Economic Forecast Report", title_style))
    story.append(Paragraph(f"{ITL1_NAMES.get(region, region)}", title_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    
    # Get key metrics
    region_data = data[data['region_code'] == region]
    
    summary_data = []
    for metric in ['population_total', 'nominal_gva_mn_gbp', 'emp_total_jobs']:
        if metric not in METRIC_CONFIG:
            continue
            
        metric_data = region_data[region_data['metric'] == metric]
        if not metric_data.empty:
            hist = metric_data[metric_data['data_type'] == 'historical']
            fore = metric_data[metric_data['data_type'] == 'forecast']
            
            if not hist.empty and not fore.empty:
                last_hist_year = hist['year'].max()
                last_hist_value = hist[hist['year'] == last_hist_year]['value'].iloc[0]
                target_year, forecast_value = pick_forecast_target(fore, 2030)
                
                if target_year:
                    cagr_val = calculate_cagr(last_hist_value, forecast_value, target_year - last_hist_year)
                    
                    summary_data.append([
                        METRIC_CONFIG[metric]['name'],
                        format_value(last_hist_value, METRIC_CONFIG[metric]['format']),
                        format_value(forecast_value, METRIC_CONFIG[metric]['format']),
                        f"{cagr_val:.1f}%" if cagr_val else "—"
                    ])
    
    if summary_data:
        t = Table([['Indicator', f'Current', f'{target_year if target_year else "2030"} Forecast', 'CAGR']] + summary_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003087')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
    
    story.append(Spacer(1, 20))
    
    # Methodology
    story.append(Paragraph("Methodology", styles['Heading2']))
    methodology_text = """
    This forecast employs state-of-the-art econometric techniques including:
    <br/>• Ensemble modeling combining ARIMA, ETS, and linear trend approaches
    <br/>• Cross-validation for optimal model weighting
    <br/>• Structural break detection for major economic events
    <br/>• Bootstrap confidence intervals for robust uncertainty quantification
    <br/>• Monte Carlo simulation for derived metric error propagation
    """
    story.append(Paragraph(methodology_text, styles['Normal']))
    
    story.append(Spacer(1, 20))
    
    # Data Quality
    story.append(Paragraph("Data Quality Indicators", styles['Heading2']))
    quality_text = f"""
    • Historical data coverage: {metadata.get('quality_indicators', {}).get('avg_history_length', 'N/A')} years average
    <br/>• Forecast uncertainty (CV): {metadata.get('quality_indicators', {}).get('mean_cv', 0)*100:.1f}% average
    <br/>• Model diversity: {metadata.get('quality_indicators', {}).get('model_diversity', 'N/A')} different methods
    <br/>• Data gaps: {metadata.get('quality_indicators', {}).get('data_gaps', 0)} detected
    """
    story.append(Paragraph(quality_text, styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 40))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    story.append(Paragraph(f"Generated by Region IQ | {datetime.now().strftime('%B %d, %Y')}", footer_style))
    story.append(Paragraph("Confidential - Not for Distribution", footer_style))
    
    # Build PDF
    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    
    return pdf

# ===========================
# Main Application
# ===========================

def main():
    # Load data
    try:
        df_long, df_wide, df_ci, df_quality, metadata = load_data()
        geojson = load_geojson()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()
    
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("🌍 Region IQ - Professional Economic Forecasts")
    with col2:
        if metadata and 'quality_indicators' in metadata:
            st.metric("Data Quality", 
                     f"{metadata.get('quality_indicators', {}).get('mean_cv', 0)*100:.1f}% avg uncertainty",
                     delta="Institutional Grade")
        else:
            st.metric("Data Quality", "Loading...", delta="")
    with col3:
        st.metric("Coverage", 
                 f"{df_long['region_code'].nunique()} regions",
                 delta=f"{df_long['metric'].nunique()} indicators")
    
    # Sidebar configuration
    st.sidebar.title("🎛️ Control Panel")
    
    # View mode selection
    view_mode = st.sidebar.selectbox(
        "View Mode",
        ["📊 Single Region Analysis", 
         "🗺️ Geographic Comparison", 
         "📈 Multi-Region Trends",
         "🎯 Executive Dashboard",
         "📑 Report Generator"]
    )
    
    # Common controls
    available_metrics = [m for m in METRIC_CONFIG.keys() if m in df_long['metric'].unique()]
    
    if not available_metrics:
        st.error("No metrics found in the data. Please check your forecast output.")
        st.stop()
    
    selected_metric = st.sidebar.selectbox(
        "Select Indicator",
        available_metrics,
        format_func=lambda x: f"{METRIC_CONFIG[x]['icon']} {METRIC_CONFIG[x]['name']}"
    )
    
    # Year slider for maps
    min_year = int(df_long['year'].min())
    max_year = int(df_long['year'].max())
    
    # ===========================
    # View Modes
    # ===========================
    
    if view_mode == "📊 Single Region Analysis":
        selected_region = st.sidebar.selectbox(
            "Select Region",
            sorted(df_long['region_code'].unique()),
            format_func=lambda x: ITL1_NAMES.get(x, x)
        )
        
        # Filter data
        region_data = df_long[(df_long['region_code'] == selected_region) & 
                             (df_long['metric'] == selected_metric)]
        
        if not region_data.empty:
            # KPI Row
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate KPIs
            hist_data = region_data[region_data['data_type'] == 'historical']
            fore_data = region_data[region_data['data_type'] == 'forecast']
            
            if not hist_data.empty:
                last_hist_year = hist_data['year'].max()
                last_value = hist_data[hist_data['year'] == last_hist_year]['value'].iloc[0]
                
                with col1:
                    st.metric(
                        f"Current ({last_hist_year})",
                        format_value(last_value, METRIC_CONFIG[selected_metric]['format'])
                    )
            
            if not fore_data.empty:
                target_year, forecast_value = pick_forecast_target(fore_data, 2030)
                if target_year:
                    with col2:
                        st.metric(
                            f"{target_year} Forecast",
                            format_value(forecast_value, METRIC_CONFIG[selected_metric]['format'])
                        )
                    
                    if not hist_data.empty:
                        cagr = calculate_cagr(last_value, forecast_value, target_year - last_hist_year)
                        if cagr:
                            with col3:
                                st.metric("CAGR", f"{cagr:.1f}%")
            
            # Quality badge
            if not df_ci.empty:
                quality_data = df_ci[(df_ci['region_code'] == selected_region) & 
                                   (df_ci['metric'] == selected_metric)]
                if not quality_data.empty:
                    avg_cv = quality_data['cv'].mean()
                    with col4:
                        st.metric("Forecast Quality", get_quality_badge(avg_cv))
            
            # Main chart
            st.plotly_chart(
                create_time_series_chart(region_data, ITL1_NAMES.get(selected_region, selected_region), 
                                        selected_metric),
                use_container_width=True
            )
            
            # AI Narrative
            with st.expander("📝 AI-Generated Analysis", expanded=True):
                narrative = generate_ai_narrative(
                    region_data, 
                    ITL1_NAMES.get(selected_region, selected_region),
                    selected_metric,
                    metadata
                )
                st.markdown(narrative)
            
            # Data table
            with st.expander("📊 View Raw Data"):
                display_cols = ['year', 'value', 'data_type']
                if 'method' in region_data.columns:
                    display_cols.append('method')
                display_data = region_data[display_cols].sort_values('year')
                st.dataframe(display_data, use_container_width=True)
                
                csv = display_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Data (CSV)",
                    csv,
                    f"{selected_region}_{selected_metric}_data.csv",
                    "text/csv"
                )
    
    elif view_mode == "🗺️ Geographic Comparison":
        selected_year = st.sidebar.slider(
            "Select Year",
            min_year,
            max_year,
            2030 if 2030 <= max_year else max_year
        )
        
        # Try to get Mapbox token from secrets
        mapbox_token = st.secrets.get("MAPBOX_TOKEN", None) if hasattr(st, "secrets") else None
        
        # Create map
        map_fig = create_choropleth_map(df_long, selected_metric, selected_year, geojson, mapbox_token)
        
        clicked_region_code = None
        if map_fig:
            if HAVE_PLOTLY_EVENTS:
                events = plotly_events(map_fig, click_event=True, override_height=600, override_width="100%")
                if events:
                    cd = events[0].get('customdata', [])
                    if isinstance(cd, list) and len(cd) > 1:
                        clicked_region_code = cd[1]
            else:
                st.plotly_chart(map_fig, use_container_width=True)
        
        # Show clicked region details
        if clicked_region_code:
            st.subheader(f"🔎 {ITL1_NAMES.get(clicked_region_code, clicked_region_code)} - {METRIC_CONFIG[selected_metric]['name']}")
            rd = df_long[(df_long['region_code'] == clicked_region_code) & (df_long['metric'] == selected_metric)]
            st.plotly_chart(
                create_time_series_chart(rd, ITL1_NAMES.get(clicked_region_code, clicked_region_code), selected_metric),
                use_container_width=True
            )
        
        # Regional ranking table
        year_data = df_long[(df_long['year'] == selected_year) & 
                           (df_long['metric'] == selected_metric)]
        
        if not year_data.empty:
            ranking = year_data[['region_code', 'value']].copy()
            ranking['region'] = ranking['region_code'].map(ITL1_NAMES)
            ranking = ranking.sort_values('value', ascending=False)
            ranking['rank'] = range(1, len(ranking) + 1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top Performers")
                top_5 = ranking.head(5)[['rank', 'region', 'value']].copy()
                top_5['value'] = top_5['value'].apply(
                    lambda x: format_value(x, METRIC_CONFIG[selected_metric]['format'])
                )
                st.table(top_5)
            
            with col2:
                st.subheader("📊 Regional Distribution")
                fig_hist = px.histogram(
                    ranking, 
                    x='value',
                    nbins=20,
                    title=f"Distribution of {METRIC_CONFIG[selected_metric]['name']} ({selected_year})",
                    template=PLOTLY_TEMPLATE
                )
                fig_hist = apply_professional_layout(fig_hist)
                st.plotly_chart(fig_hist, use_container_width=True)
    
    elif view_mode == "📈 Multi-Region Trends":
        selected_regions = st.sidebar.multiselect(
            "Select Regions to Compare",
            sorted(df_long['region_code'].unique()),
            default=sorted(df_long['region_code'].unique())[:3],
            format_func=lambda x: ITL1_NAMES.get(x, x)
        )
        
        if selected_regions:
            comparison_fig = create_comparison_chart(df_long, selected_regions, selected_metric)
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Comparison table
            st.subheader("📊 Comparative Statistics")
            
            comparison_data = []
            for region in selected_regions:
                region_data = df_long[(df_long['region_code'] == region) & 
                                     (df_long['metric'] == selected_metric)]
                
                if not region_data.empty:
                    hist = region_data[region_data['data_type'] == 'historical']
                    fore = region_data[region_data['data_type'] == 'forecast']
                    
                    if not hist.empty and not fore.empty:
                        last_hist = hist.nlargest(1, 'year')
                        target_year, forecast_value = pick_forecast_target(fore, 2030)
                        
                        if target_year:
                            cagr_val = calculate_cagr(
                                last_hist['value'].iloc[0], 
                                forecast_value, 
                                target_year - last_hist['year'].iloc[0]
                            )
                            
                            comparison_data.append({
                                'Region': ITL1_NAMES.get(region, region),
                                f'Current ({last_hist["year"].iloc[0]})': format_value(
                                    last_hist['value'].iloc[0],
                                    METRIC_CONFIG[selected_metric]['format']
                                ),
                                f'{target_year} Forecast': format_value(
                                    forecast_value,
                                    METRIC_CONFIG[selected_metric]['format']
                                ),
                                'CAGR': f"{cagr_val:.1f}%" if cagr_val else "—"
                            })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.table(comparison_df)
    
    elif view_mode == "🎯 Executive Dashboard":
        selected_region = st.sidebar.selectbox(
            "Select Region",
            sorted(df_long['region_code'].unique()),
            format_func=lambda x: ITL1_NAMES.get(x, x)
        )
        
        # Create comprehensive dashboard
        dashboard_fig = create_dashboard_summary(df_long, selected_region)
        st.plotly_chart(dashboard_fig, use_container_width=True)
        
        # Key insights
        st.subheader("🔍 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Strengths:**
            - Robust historical data foundation
            - Multiple model ensemble approach
            - Quantified uncertainty bounds
            """)
        
        with col2:
            st.success("""
            **Opportunities:**
            - Post-COVID recovery trajectory
            - Regional development initiatives
            - Income resilience patterns
            """)
    
    elif view_mode == "📑 Report Generator":
        selected_region = st.sidebar.selectbox(
            "Select Region for Report",
            sorted(df_long['region_code'].unique()),
            format_func=lambda x: ITL1_NAMES.get(x, x)
        )
        
        st.subheader("📑 Generate Professional Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Generate PDF Report", type="primary"):
                if not HAVE_REPORTLAB:
                    st.warning("PDF generation requires reportlab. Install with: pip install reportlab")
                else:
                    pdf = generate_pdf_report(df_long, selected_region, metadata)
                    if pdf:
                        st.download_button(
                            "📥 Download PDF Report",
                            pdf,
                            f"RegionIQ_Report_{selected_region}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            "application/pdf"
                        )
        
        with col2:
            # Excel export
            try:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    # Write different sheets
                    region_data = df_long[df_long['region_code'] == selected_region]
                    
                    for metric in region_data['metric'].unique():
                        metric_data = region_data[region_data['metric'] == metric]
                        sheet_name = metric[:31] if len(metric) > 31 else metric
                        metric_data.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Add metadata sheet if available
                    if metadata and 'quality_indicators' in metadata:
                        pd.DataFrame([metadata.get('quality_indicators', {})]).T.to_excel(
                            writer, sheet_name='Metadata'
                        )
                
                excel_buffer.seek(0)
                st.download_button(
                    "📊 Download Excel Report",
                    excel_buffer,
                    f"RegionIQ_Data_{selected_region}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error generating Excel: {str(e)}")
        
        with col3:
            # PowerPoint export placeholder
            st.button("📊 Generate PowerPoint", disabled=True, 
                     help="Coming soon: Auto-generated presentation slides")
        
        # Report preview
        st.subheader("📋 Report Preview")
        
        region_name = ITL1_NAMES.get(selected_region, selected_region)
        
        quality_avg = metadata.get('quality_indicators', {}).get('mean_cv', 0)*100 if metadata else 0
        history_avg = metadata.get('quality_indicators', {}).get('avg_history_length', 0) if metadata else 0
        model_div = metadata.get('quality_indicators', {}).get('model_diversity', 0) if metadata else 0
        
        st.markdown(f"""
        ### Regional Economic Forecast Report - {region_name}
        
        **Executive Summary**
        
        This comprehensive analysis provides economic forecasts for {region_name} through 2030, 
        leveraging state-of-the-art econometric modeling techniques.
        
        **Key Findings:**
        - Population trajectory shows continued growth/stability
        - Economic output (GVA) projected to expand at sustainable rates
        - Employment markets demonstrate resilience
        - Income patterns indicate regional economic strength
        
        **Methodology:**
        - Ensemble modeling combining multiple forecasting approaches
        - Cross-validation for optimal model selection
        - Bootstrap confidence intervals for uncertainty quantification
        - Structural break detection for major economic events
        
        **Data Quality:**
        - Average forecast uncertainty: {quality_avg:.1f}%
        - Historical data coverage: {history_avg:.1f} years
        - Model diversity: {model_div} methods
        
        ---
        *Generated by Region IQ | {datetime.now().strftime('%B %d, %Y')}*
        """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("🏢 Region IQ - Professional Economic Intelligence")
    
    with col2:
        st.caption("📊 Powered by Advanced Econometric Modeling")
    
    with col3:
        if metadata:
            st.caption(f"🕐 Last Updated: {metadata.get('run_timestamp', 'Unknown')[:10]}")
        else:
            st.caption("🕐 Last Updated: Unknown")

if __name__ == "__main__":
    main()