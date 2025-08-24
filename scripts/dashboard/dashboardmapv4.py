
"""
Region IQ - Dashboard V5
=========================
Regional economic intelligence platform
"""

# ABSOLUTE FIRST THING - Import streamlit and set page config
import streamlit as st

st.set_page_config(
    page_title="Region IQ",
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

# ===========================
# Professional Theme & Typography
# ===========================

# Inject CSS with modern SaaS color palette
st.markdown("""
    <style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global theme */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background: #0E1117 !important;
        color: #E7EBF0 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-feature-settings: "tnum" 1;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #151922 !important;
        border-right: 1px solid #262B37;
    }
    
    /* Sidebar labels */
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] label {
        color: #9AA4B2 !important;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    /* Headers */
    h1 {
        color: #E7EBF0 !important;
        font-weight: 800;
        font-size: 2.25rem;
        border-bottom: 2px solid #262B37;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
    }
    
    h2 {
        color: #E7EBF0 !important;
        font-weight: 700;
        font-size: 1.5rem;
        margin-top: 2rem;
        letter-spacing: -0.01em;
    }
    
    h3 {
        color: #E7EBF0 !important;
        font-weight: 600;
        font-size: 1.125rem;
    }
    
    /* Metric labels (grey, muted) */
    [data-testid="stMetricLabel"] {
        color: #9AA4B2 !important;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    /* Metric values (white, prominent) */
    [data-testid="stMetricValue"] {
        color: #E7EBF0 !important;
        font-weight: 700;
        font-size: 2rem;
        font-feature-settings: "tnum" 1;
    }
    
    /* Metric deltas (accent color for highlights) */
    [data-testid="stMetricDelta"] {
        color: #FF7849 !important;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    /* Buttons - subtle until hover */
    .stDownloadButton button, .stButton button {
        background: #262B37;
        color: #E7EBF0;
        border: 1px solid #363C49;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.2s ease;
    }
    
    .stDownloadButton button:hover, .stButton button:hover {
        background: #FF7849;
        border-color: #FF7849;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255, 120, 73, 0.2);
    }
    
    /* Tables */
    .stTable {
        background: #151922;
        border-radius: 8px;
        border: 1px solid #262B37;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: #151922;
        border: 1px solid #262B37;
        border-radius: 6px;
        font-weight: 600;
        color: #E7EBF0;
    }
    
    /* Info/Alert boxes */
    .stAlert {
        background: #151922;
        border-left: 3px solid #FF7849;
        border-radius: 4px;
        color: #E7EBF0;
    }
    
    .stInfo {
        background: rgba(58, 163, 255, 0.1);
        border-left: 3px solid #3AA3FF;
        color: #E7EBF0;
    }
    
    .stSuccess {
        background: rgba(0, 194, 168, 0.1);
        border-left: 3px solid #00C2A8;
        color: #E7EBF0;
    }
    
    /* Logo - subtle */
    .logo-container {
        position: fixed;
        top: 1.5rem;
        right: 2rem;
        z-index: 999;
    }
    
    .logo-text {
        font-size: 0.875rem;
        font-weight: 700;
        color: #9AA4B2;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    
    /* Dividers */
    hr {
        border-color: #262B37 !important;
        margin: 2rem 0;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #9AA4B2 !important;
    }
    
    /* Caption text */
    .caption {
        color: #9AA4B2 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Add subtle logo
st.markdown("""
    <div class="logo-container">
        <span class="logo-text">Region IQ</span>
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

# ITL1 code mapping (handles both E-codes and ITL codes)
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
    "TLN": "Northern Ireland",
    # E-code mappings
    "E12000001": "North East",
    "E12000002": "North West",
    "E12000003": "Yorkshire & Humber",
    "E12000004": "East Midlands",
    "E12000005": "West Midlands",
    "E12000006": "East of England",
    "E12000007": "London",
    "E12000008": "South East",
    "E12000009": "South West",
    "W92000004": "Wales",
    "S92000003": "Scotland",
    "N92000002": "Northern Ireland"
}

# Region code mappings (E-code to ITL)
REGION_CODE_MAP = {
    "E12000001": "TLC",
    "E12000002": "TLD",
    "E12000003": "TLE",
    "E12000004": "TLF",
    "E12000005": "TLG",
    "E12000006": "TLH",
    "E12000007": "TLI",
    "E12000008": "TLJ",
    "E12000009": "TLK",
    "W92000004": "TLL",
    "S92000003": "TLM",
    "N92000002": "TLN"
}

# Plotly theme
PLOTLY_TEMPLATE = "plotly_dark"

# Chart colors - Modern SaaS palette
HIST_COLOR = "#3AA3FF"   # Soft blue for historical
FORE_COLOR = "#FF7849"   # Modern orange for forecast (softer than pure orange)
CONF_FILL = "rgba(255, 120, 73, 0.12)"  # 12% opacity for confidence intervals
RECESSION_FILL = "rgba(156, 163, 175, 0.08)"  # Very subtle grey for recessions

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
    
    # Standardize region codes - convert all E-codes to ITL codes
    df_long['original_region_code'] = df_long['region_code']
    df_long['region_code'] = df_long['region_code'].apply(
        lambda x: REGION_CODE_MAP.get(x, x)
    )
    
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
    
    **Latest Official Data (ONS/NOMIS)**: As of {last_hist_year}, {region}'s {metric_name.lower()} stood at {format_value(last_hist_value, metric_info.get('format', 'number'))} {metric_info.get('unit', '')}.
    
    **Region IQ Forecast**: Our models project {metric_name.lower()} will reach {format_value(forecast_value, metric_info.get('format', 'number'))} {metric_info.get('unit', '')} by {target_year}, representing a compound annual growth rate of {cagr:.1f}%.
    
    **Key Drivers**: This trajectory reflects post-COVID recovery dynamics, structural economic shifts, and regional development patterns. The forecast accounts for historical volatility and structural breaks including the 2008 financial crisis and 2020 pandemic impacts.
    
    **Data Sources**: Historical data sourced from ONS (Office for National Statistics) and NOMIS official labour market statistics. Forecasts generated by Region IQ's ensemble econometric models combining ARIMA, ETS, and linear trend approaches based on {hist_data['year'].nunique()} years of historical data.
    """
    
    return narrative

# ===========================
# Visualization Functions
# ===========================

def create_time_series_chart(df, region_name, metric, region_code, show_confidence=True):
    """Create professional time series chart with forecast"""
    
    metric_info = METRIC_CONFIG.get(metric, {})
    
    # Filter data inside the function (region codes are now standardized to ITL)
    data = df[(df['region_code'] == region_code) & (df['metric'] == metric)]
    
    if data.empty:
        return go.Figure()  # Return blank figure if no data
    
    # Split historical and forecast
    hist = data[data['data_type'] == 'historical'].sort_values('year')
    fore = data[data['data_type'] == 'forecast'].sort_values('year')
    
    # Bridge the gap between historical and forecast
    if not hist.empty and not fore.empty:
        last_hist_year = hist['year'].max()
        last_hist_val = hist.loc[hist['year'] == last_hist_year, 'value'].iloc[0]
        
        # If forecast starts after history, insert a bridge row
        if fore['year'].min() > last_hist_year:
            bridge_row = pd.DataFrame({
                'year': [last_hist_year],
                'value': [last_hist_val],
                'data_type': ['forecast'],
                'region_code': [region_code],
                'metric': [metric]
            })
            # Add confidence intervals if they exist
            if 'ci_lower' in fore.columns and not fore['ci_lower'].isna().all():
                bridge_row['ci_lower'] = hist.loc[hist['year'] == last_hist_year, 'value'].iloc[0]
                bridge_row['ci_upper'] = hist.loc[hist['year'] == last_hist_year, 'value'].iloc[0]
            
            fore = pd.concat([bridge_row, fore], ignore_index=True).sort_values('year')
    
    fig = go.Figure()
    
    # Historical line (ONS/NOMIS data)
    fig.add_trace(go.Scatter(
        x=hist['year'],
        y=hist['value'],
        mode='lines+markers',
        name='Historical (ONS/NOMIS)',
        line=dict(color=HIST_COLOR, width=2.5),
        marker=dict(size=5, color=HIST_COLOR)
    ))
    
    # Forecast line (Region IQ)
    if not fore.empty:
        fig.add_trace(go.Scatter(
            x=fore['year'],
            y=fore['value'],
            mode='lines+markers',
            name='Forecast (Region IQ)',
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
    
    # Add recession shading (very subtle)
    fig.add_vrect(x0=2008, x1=2009, fillcolor=RECESSION_FILL, line_width=0)
    fig.add_vrect(x0=2020, x1=2021, fillcolor=RECESSION_FILL, line_width=0)
    
    # Add vertical line at forecast start
    if not hist.empty and not fore.empty:
        last_hist_year = hist['year'].max()
        fig.add_vline(x=last_hist_year, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    
    # Update layout
    fig = apply_professional_layout(fig, f"{region_name} - {metric_info.get('name', metric)}")
    
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
    
    colors = ['#3AA3FF', '#FF7849', '#00C2A8', '#635BFF', '#F36D5A', '#17B897', '#8B5CF6', '#EC4899']
    
    for i, region in enumerate(regions):
        region_data = data[(data['region_code'] == region) & (data['metric'] == metric)]
        
        if not region_data.empty:
            # Historical
            hist = region_data[region_data['data_type'] == 'historical'].sort_values('year')
            fore = region_data[region_data['data_type'] == 'forecast'].sort_values('year')
            
            # Bridge the gap between historical and forecast
            if not hist.empty and not fore.empty:
                last_hist_year = hist['year'].max()
                last_hist_val = hist.loc[hist['year'] == last_hist_year, 'value'].iloc[0]
                
                # If forecast starts after history, insert a bridge row
                if fore['year'].min() > last_hist_year:
                    bridge_row = pd.DataFrame({
                        'year': [last_hist_year],
                        'value': [last_hist_val],
                        'data_type': ['forecast'],
                        'region_code': [region],
                        'metric': [metric]
                    })
                    fore = pd.concat([bridge_row, fore], ignore_index=True).sort_values('year')
            
            # Plot historical
            if not hist.empty:
                fig.add_trace(go.Scatter(
                    x=hist['year'],
                    y=hist['value'],
                    mode='lines',
                    name=f"{ITL1_NAMES.get(region, region)} (ONS/NOMIS)",
                    line=dict(color=colors[i % len(colors)], width=2.5),
                    legendgroup=region
                ))
            
            # Plot forecast (with bridge)
            if not fore.empty:
                fig.add_trace(go.Scatter(
                    x=fore['year'],
                    y=fore['value'],
                    mode='lines',
                    name=f"{ITL1_NAMES.get(region, region)} (Region IQ)",
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
            
            # Bridge the gap between historical and forecast
            if not hist.empty and not fore.empty:
                last_hist_year = hist['year'].max()
                last_hist_val = hist.loc[hist['year'] == last_hist_year, 'value'].iloc[0]
                
                # If forecast starts after history, insert a bridge row
                if fore['year'].min() > last_hist_year:
                    bridge_row = pd.DataFrame({
                        'year': [last_hist_year],
                        'value': [last_hist_val]
                    })
                    fore = pd.concat([bridge_row, fore], ignore_index=True).sort_values('year')
            
            # Plot historical
            if not hist.empty:
                fig.add_trace(
                    go.Scatter(x=hist['year'], y=hist['value'], mode='lines',
                             line=dict(color=HIST_COLOR, width=2),
                             showlegend=False),
                    row=row, col=col
                )
            
            # Plot forecast (with bridge)
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
        t = Table([['Indicator', f'ONS/NOMIS ({last_hist_year})', f'Region IQ {target_year if target_year else "2030"}', 'Projected CAGR']] + summary_data)
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
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🌍 Region IQ")
    with col2:
        st.metric("Coverage", 
                 f"{df_long['region_code'].nunique()} regions",
                 delta=f"{df_long['metric'].nunique()} indicators")
    
    # Sidebar configuration
    st.sidebar.title("🎛️ Control Panel")
    
    # Debug section
    if st.sidebar.checkbox("🔎 Debug Data Check"):
        st.sidebar.write("**Unique metrics in CSV:**")
        st.sidebar.write(df_long['metric'].unique().tolist())
        
        with st.expander("📊 Full Debug Information", expanded=True):
            st.subheader("Data Structure Analysis")
            
            # Show columns
            st.write("**Columns in df_long:**", df_long.columns.tolist())
            
            # Show unique metrics
            st.write("**Unique metrics found:**", df_long['metric'].unique().tolist())
            
            # Show data types if exists
            if 'data_type' in df_long.columns:
                st.write("**Unique data_types:**", df_long['data_type'].unique().tolist())
            else:
                st.error("⚠️ 'data_type' column is MISSING!")
            
            # Check each metric
            st.subheader("Metric-by-Metric Analysis")
            
            metrics_to_check = [
                'population_total',
                'gdhi_total_mn_gbp', 
                'gdhi_per_head_gbp',
                'nominal_gva_mn_gbp',
                'chained_gva_mn_gbp',
                'emp_total_jobs',
                'employment_rate',
                'income_per_worker_gbp'
            ]
            
            for metric in metrics_to_check:
                st.write(f"\n**Checking: {metric}**")
                metric_data = df_long[df_long['metric'] == metric]
                
                if metric_data.empty:
                    st.warning(f"❌ No data found for {metric}")
                else:
                    st.success(f"✅ Found {len(metric_data)} rows for {metric}")
                    
                    # Show data type breakdown
                    if 'data_type' in metric_data.columns:
                        type_counts = metric_data['data_type'].value_counts()
                        st.write(f"   - Data types: {type_counts.to_dict()}")
                    
                    # Show year range
                    st.write(f"   - Year range: {metric_data['year'].min()} to {metric_data['year'].max()}")
                    
                    # Show sample
                    st.write("   Sample rows:")
                    st.dataframe(metric_data.head(5))
            
            # Show raw sample
            st.subheader("Raw Data Sample (first 20 rows)")
            st.dataframe(df_long.head(20))
            
            # Potential fixes
            st.subheader("🛠️ Diagnostic Results")
            
            # Check for common issues
            issues = []
            
            if 'data_type' not in df_long.columns:
                issues.append("• **CRITICAL**: 'data_type' column is missing - charts cannot distinguish historical vs forecast")
            
            expected_metrics = set(METRIC_CONFIG.keys())
            actual_metrics = set(df_long['metric'].unique())
            missing_metrics = expected_metrics - actual_metrics
            
            if missing_metrics:
                issues.append(f"• **Missing metrics**: {list(missing_metrics)}")
            
            unexpected_metrics = actual_metrics - expected_metrics
            if unexpected_metrics:
                issues.append(f"• **Unexpected metrics in CSV**: {list(unexpected_metrics)} - update METRIC_CONFIG to include these")
            
            if issues:
                st.error("**Issues Found:**")
                for issue in issues:
                    st.write(issue)
            else:
                st.success("✅ No obvious issues found - data structure looks correct")
    
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
            # KPI Row with benchmarking
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate KPIs
            hist_data = region_data[region_data['data_type'] == 'historical']
            fore_data = region_data[region_data['data_type'] == 'forecast']
            
            if not hist_data.empty:
                last_hist_year = hist_data['year'].max()
                last_value = hist_data[hist_data['year'] == last_hist_year]['value'].iloc[0]
                
                with col1:
                    st.metric(
                        f"Latest ONS/NOMIS Data ({last_hist_year})",
                        format_value(last_value, METRIC_CONFIG[selected_metric]['format'])
                    )
                
                # Calculate UK average and ranking
                all_regions_data = df_long[
                    (df_long['metric'] == selected_metric) & 
                    (df_long['year'] == last_hist_year) & 
                    (df_long['data_type'] == 'historical')
                ]
                
                if not all_regions_data.empty:
                    uk_avg = all_regions_data['value'].mean()
                    region_rank = all_regions_data.sort_values('value', ascending=False).reset_index(drop=True)
                    region_rank['rank'] = region_rank.index + 1
                    current_rank = region_rank[region_rank['region_code'] == selected_region]['rank'].iloc[0] if not region_rank[region_rank['region_code'] == selected_region].empty else None
                    
                    with col2:
                        # Show UK average comparison
                        diff_from_avg = ((last_value - uk_avg) / uk_avg) * 100
                        delta_color = "normal" if diff_from_avg >= 0 else "inverse"
                        st.metric(
                            f"UK Average ({last_hist_year})",
                            format_value(uk_avg, METRIC_CONFIG[selected_metric]['format']),
                            delta=f"{diff_from_avg:+.1f}% vs avg",
                            delta_color=delta_color
                        )
            
            if not fore_data.empty:
                target_year, forecast_value = pick_forecast_target(fore_data, 2030)
                if target_year:
                    with col3:
                        st.metric(
                            f"{target_year} Region IQ Forecast",
                            format_value(forecast_value, METRIC_CONFIG[selected_metric]['format'])
                        )
                    
                    if not hist_data.empty:
                        cagr = calculate_cagr(last_value, forecast_value, target_year - last_hist_year)
                        if cagr:
                            total_regions = len(all_regions_data) if 'all_regions_data' in locals() else df_long['region_code'].nunique()
                            with col4:
                                if 'current_rank' in locals() and current_rank:
                                    st.metric("Position & Growth", 
                                             f"#{current_rank} of {total_regions}",
                                             delta=f"CAGR: {cagr:.1f}%")
                                else:
                                    st.metric("Projected Growth", 
                                             f"CAGR: {cagr:.1f}%")
            
            # Main chart
            st.plotly_chart(
                create_time_series_chart(
                    df_long,  # Pass full dataset
                    ITL1_NAMES.get(selected_region, selected_region),
                    selected_metric,
                    selected_region  # Pass region code
                ),
                use_container_width=True
            )
            
            # Peer Benchmarking Section
            with st.expander("📊 Peer Region Comparison", expanded=True):
                # Get latest year data for all regions
                if not hist_data.empty:
                    latest_year = hist_data['year'].max()
                    peer_data = df_long[
                        (df_long['metric'] == selected_metric) & 
                        (df_long['year'] == latest_year) & 
                        (df_long['data_type'] == 'historical')
                    ].copy()
                    
                    if not peer_data.empty:
                        peer_data['region_name'] = peer_data['region_code'].map(ITL1_NAMES)
                        peer_data = peer_data.sort_values('value', ascending=False)
                        peer_data['rank'] = range(1, len(peer_data) + 1)
                        
                        # Create two columns for the comparison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Show top 5 performers
                            st.markdown("**Top Performing Regions**")
                            top_regions = peer_data.head(5)[['rank', 'region_name', 'value']].copy()
                            top_regions['value'] = top_regions['value'].apply(
                                lambda x: format_value(x, METRIC_CONFIG[selected_metric]['format'])
                            )
                            # Highlight current region if in top 5
                            top_regions['region_name'] = top_regions.apply(
                                lambda row: f"**{row['region_name']}**" if row['region_name'] == ITL1_NAMES.get(selected_region, selected_region) else row['region_name'],
                                axis=1
                            )
                            st.dataframe(top_regions, hide_index=True, use_container_width=True)
                        
                        with col2:
                            # Show regional statistics
                            st.markdown("**Statistical Summary**")
                            uk_mean = peer_data['value'].mean()
                            uk_median = peer_data['value'].median()
                            uk_std = peer_data['value'].std()
                            current_value = peer_data[peer_data['region_code'] == selected_region]['value'].iloc[0] if not peer_data[peer_data['region_code'] == selected_region].empty else None
                            
                            if current_value:
                                z_score = (current_value - uk_mean) / uk_std if uk_std > 0 else 0
                                percentile = (peer_data[peer_data['value'] <= current_value].shape[0] / peer_data.shape[0]) * 100
                                
                                # Format the statistical summary similar to OE style
                                metric_format = METRIC_CONFIG[selected_metric]['format']
                                
                                st.markdown(f"""
                                **UK Mean:** {format_value(uk_mean, metric_format)}
                                
                                **UK Median:** {format_value(uk_median, metric_format)}
                                
                                **{ITL1_NAMES.get(selected_region, selected_region)}:** {format_value(current_value, metric_format)}
                                
                                **Percentile:** {percentile:.0f}th percentile
                                
                                **Std Devs from Mean:** {z_score:+.2f}σ
                                """)
                                
                                # Add note about totals if it's a per-capita metric
                                if any(term in selected_metric for term in ['per_head', 'per_capita', 'per_job', 'per_worker']):
                                    st.info("💡 **Note**: Showing per-capita values. Regional totals require population weighting - contact Region IQ for aggregate analysis.")
            
            # AI Narrative
            with st.expander("📝 AI-Generated Analysis", expanded=True):
                narrative = generate_ai_narrative(
                    region_data, 
                    ITL1_NAMES.get(selected_region, selected_region),
                    selected_metric,
                    metadata
                )
                st.markdown(narrative)
            
            # Professional Data Table - Simple Combined View
            with st.expander("📊 View Data Table"):
                # Prepare data for wide format
                hist_table = region_data[region_data['data_type'] == 'historical'][['year', 'value']].sort_values('year')
                fore_table = region_data[region_data['data_type'] == 'forecast'][['year', 'value']].sort_values('year')
                
                # Create combined wide format table
                if not hist_table.empty or not fore_table.empty:
                    # Create year range from min to max
                    all_years = []
                    if not hist_table.empty:
                        all_years.extend(hist_table['year'].tolist())
                    if not fore_table.empty:
                        all_years.extend(fore_table['year'].tolist())
                    
                    year_range = range(int(min(all_years)), int(max(all_years)) + 1)
                    
                    # Create combined dataframe
                    combined_df = pd.DataFrame({'Year': year_range})
                    
                    # Add historical values
                    if not hist_table.empty:
                        hist_merge = hist_table.copy()
                        hist_merge.columns = ['Year', 'Historical (ONS/NOMIS)']
                        hist_merge['Year'] = hist_merge['Year'].astype(int)
                        combined_df = combined_df.merge(hist_merge, on='Year', how='left')
                    
                    # Add forecast values
                    if not fore_table.empty:
                        fore_merge = fore_table.copy()
                        fore_merge.columns = ['Year', 'Forecast (Region IQ)']
                        fore_merge['Year'] = fore_merge['Year'].astype(int)
                        combined_df = combined_df.merge(fore_merge, on='Year', how='left')
                    
                    # Format all numeric columns
                    for col in combined_df.columns:
                        if col != 'Year':
                            combined_df[col] = combined_df[col].apply(
                                lambda x: format_value(x, METRIC_CONFIG[selected_metric]['format']) if pd.notna(x) else "—"
                            )
                    
                    # Display the table
                    st.dataframe(combined_df.set_index('Year'), use_container_width=True)
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # CSV download
                        csv_data = combined_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv_data,
                            f"{selected_region}_{selected_metric}_data.csv",
                            "text/csv"
                        )
                    
                    with col2:
                        # Excel download
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            combined_df.to_excel(writer, sheet_name='Data', index=False)
                        excel_buffer.seek(0)
                        st.download_button(
                            "📥 Download Excel",
                            excel_buffer,
                            f"{selected_region}_{selected_metric}_data.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    # Source caption
                    st.caption("**Data Sources**: Historical data from ONS/NOMIS | Forecasts by Region IQ")
    
    elif view_mode == "🗺️ Geographic Comparison":
        selected_year = st.sidebar.slider(
            "Select Year",
            min_year,
            max_year,
            2030 if 2030 <= max_year else max_year
        )
        
        # Try to get Mapbox token from secrets
        # mapbox_token = st.secrets.get("MAPBOX_TOKEN", None) if hasattr(st, "secrets") else None
        
        # Create map
        map_fig = create_choropleth_map(df_long, selected_metric, selected_year, geojson)
        
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
            st.plotly_chart(
                create_time_series_chart(
                    df_long,  # Pass full dataset
                    ITL1_NAMES.get(clicked_region_code, clicked_region_code),
                    selected_metric,
                    clicked_region_code  # Pass region code
                ),
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
                                f'ONS/NOMIS ({last_hist["year"].iloc[0]})': format_value(
                                    last_hist['value'].iloc[0],
                                    METRIC_CONFIG[selected_metric]['format']
                                ),
                                f'Region IQ {target_year}': format_value(
                                    forecast_value,
                                    METRIC_CONFIG[selected_metric]['format']
                                ),
                                'Projected CAGR': f"{cagr_val:.1f}%" if cagr_val else "—"
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
        st.caption("🏢 Region IQ | Economic Intelligence")
    
    with col2:
        st.caption("📊 Data: ONS/NOMIS | Forecasts: Region IQ")
    
    with col3:
        if metadata:
            st.caption(f"🕐 Last Updated: {metadata.get('run_timestamp', 'Unknown')[:10]}")
        else:
            st.caption("🕐 Last Updated: Unknown")

if __name__ == "__main__":
    main()