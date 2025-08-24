#!/usr/bin/env python3
"""
Data Diagnostic Script for Region IQ
=====================================
Run this to check if your forecast data has the correct structure
"""

import pandas as pd
from pathlib import Path
import json

# Configuration
FORECAST_LONG = Path("data/forecast/forecast_v3_long.csv")
FORECAST_WIDE = Path("data/forecast/forecast_v3_wide.csv")
CONFIDENCE_INTERVALS = Path("data/forecast/confidence_intervals_v3.csv")

# Expected metrics from dashboard
EXPECTED_METRICS = [
    'population_total',
    'gdhi_total_mn_gbp',
    'gdhi_per_head_gbp',
    'nominal_gva_mn_gbp',
    'chained_gva_mn_gbp',
    'emp_total_jobs',
    'employment_rate',
    'income_per_worker_gbp'
]

def check_file_exists(filepath):
    """Check if file exists"""
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False
    print(f"✅ File exists: {filepath}")
    return True

def diagnose_forecast_data():
    """Run comprehensive diagnostics on forecast data"""
    
    print("=" * 60)
    print("REGION IQ DATA DIAGNOSTIC")
    print("=" * 60)
    
    # Check file existence
    if not check_file_exists(FORECAST_LONG):
        print("\n⚠️  Cannot proceed without forecast_v3_long.csv")
        return
    
    # Load data
    print("\n📊 Loading data...")
    df = pd.read_csv(FORECAST_LONG)
    print(f"   Loaded {len(df)} rows")
    
    # Check columns
    print("\n📋 Column Analysis:")
    print(f"   Columns found: {df.columns.tolist()}")
    
    required_cols = ['year', 'value', 'region_code', 'metric', 'data_type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"   ❌ MISSING REQUIRED COLUMNS: {missing_cols}")
        print("      This will cause charts to fail!")
    else:
        print("   ✅ All required columns present")
    
    # Check data_type values
    if 'data_type' in df.columns:
        print(f"\n📊 Data Types Found:")
        print(f"   {df['data_type'].unique().tolist()}")
        
        if not {'historical', 'forecast'}.issubset(set(df['data_type'].unique())):
            print("   ⚠️  Warning: Expected 'historical' and 'forecast' in data_type")
    
    # Check metrics
    print(f"\n📈 Metrics Analysis:")
    actual_metrics = df['metric'].unique().tolist() if 'metric' in df.columns else []
    print(f"   Metrics in CSV: {actual_metrics}")
    
    # Compare with expected
    print(f"\n🔍 Comparing with Dashboard Configuration:")
    
    for metric in EXPECTED_METRICS:
        if metric in actual_metrics:
            metric_data = df[df['metric'] == metric]
            
            # Check for both historical and forecast
            if 'data_type' in df.columns:
                has_hist = 'historical' in metric_data['data_type'].values
                has_fore = 'forecast' in metric_data['data_type'].values
                
                if has_hist and has_fore:
                    print(f"   ✅ {metric}: {len(metric_data)} rows (historical + forecast)")
                elif has_hist:
                    print(f"   ⚠️  {metric}: Only historical data found")
                elif has_fore:
                    print(f"   ⚠️  {metric}: Only forecast data found")
                else:
                    print(f"   ❌ {metric}: No valid data_type values")
            else:
                print(f"   ⚠️  {metric}: Found {len(metric_data)} rows but no data_type column")
        else:
            print(f"   ❌ {metric}: NOT FOUND in CSV")
    
    # Check for unexpected metrics
    unexpected = set(actual_metrics) - set(EXPECTED_METRICS)
    if unexpected:
        print(f"\n⚠️  Unexpected metrics in CSV (not in dashboard config):")
        for m in unexpected:
            print(f"   - {m}")
    
    # Sample data for working metrics
    print("\n📊 Sample Data for Working Metrics (GDHI):")
    if 'gdhi_total_mn_gbp' in actual_metrics:
        sample = df[df['metric'] == 'gdhi_total_mn_gbp'].head(5)
        print(sample.to_string())
    
    # Sample data for non-working metric
    print("\n📊 Sample Data for Non-Working Metric (GVA):")
    if 'nominal_gva_mn_gbp' in actual_metrics:
        sample = df[df['metric'] == 'nominal_gva_mn_gbp'].head(5)
        print(sample.to_string())
    elif 'gva' in ' '.join(actual_metrics).lower():
        # Try to find any GVA-related metric
        gva_metrics = [m for m in actual_metrics if 'gva' in m.lower()]
        if gva_metrics:
            print(f"   Found GVA variants: {gva_metrics}")
            print(f"   👉 Update METRIC_CONFIG to use '{gva_metrics[0]}' instead of 'nominal_gva_mn_gbp'")
    
    # Region check
    if 'region_code' in df.columns:
        print(f"\n🌍 Regions Found:")
        print(f"   {df['region_code'].unique().tolist()}")
    
    # Year range check
    if 'year' in df.columns:
        print(f"\n📅 Year Range:")
        print(f"   Min: {df['year'].min()}, Max: {df['year'].max()}")
        
        # Check for gaps
        if 'data_type' in df.columns:
            hist_years = df[df['data_type'] == 'historical']['year'].unique()
            fore_years = df[df['data_type'] == 'forecast']['year'].unique()
            
            if len(hist_years) > 0 and len(fore_years) > 0:
                gap = min(fore_years) - max(hist_years)
                if gap > 1:
                    print(f"   ⚠️  Gap detected: {gap} years between historical and forecast")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    
    # Summary recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if missing_cols:
        print("1. ❗ Add missing columns to your forecast export:")
        for col in missing_cols:
            print(f"   - {col}")
    
    if 'data_type' not in df.columns:
        print("2. ❗ Add 'data_type' column with values 'historical' or 'forecast'")
    
    missing_in_csv = set(EXPECTED_METRICS) - set(actual_metrics)
    if missing_in_csv:
        print("3. ❗ These metrics need to be added to your forecast pipeline:")
        for m in missing_in_csv:
            print(f"   - {m}")
    
    if unexpected:
        print("4. 💡 Consider updating METRIC_CONFIG in dashboard to include:")
        for m in unexpected:
            print(f"   - '{m}': {{ 'name': '...', 'unit': '...', ... }}")

def quick_fix_check():
    """Quick check for the most common issue"""
    print("\n🚀 QUICK FIX CHECK:")
    
    if not FORECAST_LONG.exists():
        print("❌ forecast_v3_long.csv not found")
        return
    
    df = pd.read_csv(FORECAST_LONG)
    
    # Most common issue: metric name mismatch
    actual_metrics = df['metric'].unique().tolist() if 'metric' in df.columns else []
    
    print("\nMetric Name Mapping Suggestions:")
    
    # Common variations
    mappings = {
        'gva': 'nominal_gva_mn_gbp',
        'gva_nominal': 'nominal_gva_mn_gbp',
        'gva_chained': 'chained_gva_mn_gbp',
        'population': 'population_total',
        'employment': 'emp_total_jobs',
        'jobs': 'emp_total_jobs',
        'gdhi': 'gdhi_total_mn_gbp',
        'gdhi_per_capita': 'gdhi_per_head_gbp'
    }
    
    for actual in actual_metrics:
        for pattern, expected in mappings.items():
            if pattern in actual.lower() and expected not in actual_metrics:
                print(f"   '{actual}' → rename to '{expected}' in CSV")
                print(f"   OR update METRIC_CONFIG: '{actual}' instead of '{expected}'")

if __name__ == "__main__":
    diagnose_forecast_data()
    quick_fix_check()
    
    print("\n✅ Diagnostic complete! Check output above for issues.")
    print("📝 To fix: Either update your forecast export or modify METRIC_CONFIG in dashboard.")