#!/usr/bin/env python3
"""
Clean ONS GVA data - extract ITL1 totals for nominal and chained volume measures.
Matches the output format of income_gdhi_itl1.py

Expected input: gva_YYYYMMDD_*.xlsx from scrape_gva.py
Outputs:
  - gva_ITL1_long.csv (tidy format)
  - gva_ITL1_wide.csv (pivot format)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
import logging

# Setup paths
IN_DIR = Path("data/raw/gva")
OUT_DIR = Path("data/clean/gva")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LONG = OUT_DIR / "gva_ITL1_long.csv"
OUT_WIDE = OUT_DIR / "gva_ITL1_wide.csv"

# Find latest GVA file
gva_files = sorted(IN_DIR.glob("gva_*.xlsx"))
if not gva_files:
    raise FileNotFoundError(f"No GVA files found in {IN_DIR}")
IN = gva_files[-1]  # Use most recent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ITL1 region mapping (standardize names to match income cleaner)
ITL1_REGIONS = {
    'TLC': 'North East',
    'TLD': 'North West', 
    'TLE': 'Yorkshire & Humber',  # Note: & not "and The"
    'TLF': 'East Midlands',
    'TLG': 'West Midlands',
    'TLH': 'East of England',
    'TLI': 'London',
    'TLJ': 'South East',
    'TLK': 'South West',
    'TLL': 'Wales',
    'TLM': 'Scotland',
    'TLN': 'Northern Ireland'
}

def clean_numeric(series):
    """Clean numeric columns, handling ONS special values."""
    return pd.to_numeric(
        series.astype(str).str.replace('[c]', '', regex=False)
                          .str.replace('..', '', regex=False)
                          .str.replace('np', '', regex=False)
                          .str.strip(),
        errors='coerce'
    )

def extract_itl1_data(sheet_df, measure_name, is_index=False):
    """
    Extract ITL1 total GVA from a Table 1 sheet.
    
    Args:
        sheet_df: Raw dataframe from Excel sheet
        measure_name: 'nominal_gva_mn_gbp' or 'chained_gva_mn_gbp'
        is_index: True if this is an index sheet (Table 1a), False for money values (Table 1b)
    
    Returns:
        DataFrame with columns: region, region_code, year, metric, value
    """
    # The structure is:
    # Row 0: Title
    # Row 1: Headers (ITL code, Region name, SIC07 code, Industry, then years)
    # Row 2+: Data
    
    # Set headers from row 1
    sheet_df.columns = sheet_df.iloc[1].fillna('').astype(str)
    
    # Drop the header rows and reset index
    sheet_df = sheet_df.iloc[2:].reset_index(drop=True)
    
    # Clean column names
    sheet_df.columns = [str(col).strip() for col in sheet_df.columns]
    
    # Identify year columns (numeric values that look like years)
    year_cols = []
    for col in sheet_df.columns:
        try:
            # Convert to float first to handle '2023.0' format
            year_val = float(col)
            if 1990 <= year_val <= 2030:
                year_cols.append(col)
        except (ValueError, TypeError):
            continue
    
    if not year_cols:
        logger.warning(f"No year columns found for {measure_name}")
        return pd.DataFrame()
    
    logger.info(f"  Found {len(year_cols)} year columns: {year_cols[0]} to {year_cols[-1]}")
    
    # Rename the first few columns for consistency
    col_mapping = {}
    for i, col in enumerate(sheet_df.columns[:4]):
        if 'ITL' in col and i == 0:  # First column is ITL code
            col_mapping[col] = 'itl_code'
        elif 'Region' in col and i == 1:  # Second column is region name
            col_mapping[col] = 'region_name'
        elif 'SIC' in col and 'code' in col and i == 2:  # Third column is SIC code
            col_mapping[col] = 'sic_code'
        elif ('SIC' in col and 'description' in col) or ('Industry' in col) or i == 3:  # Fourth column is industry description
            col_mapping[col] = 'industry'
    
    sheet_df = sheet_df.rename(columns=col_mapping)
    
    # Filter for ITL1 regions (TLC-TLN) or UK total
    itl1_mask = sheet_df['itl_code'].isin(list(ITL1_REGIONS.keys()) + ['UK'])
    itl1_df = sheet_df[itl1_mask].copy()
    
    logger.info(f"  Found {len(itl1_df)} ITL1 region rows (including UK)")
    
    # Get total rows (SIC code 'Total' or industry contains 'Total')
    total_mask = (
        (itl1_df['sic_code'].astype(str).str.lower() == 'total') |
        (itl1_df.get('industry', pd.Series()).str.contains('Total', case=False, na=False))
    )
    total_df = itl1_df[total_mask].copy()
    
    if total_df.empty:
        logger.warning(f"No total GVA rows found for {measure_name}")
        return pd.DataFrame()
    
    logger.info(f"  Found {len(total_df)} total GVA rows")
    
    # Keep only relevant columns
    keep_cols = ['itl_code'] + year_cols
    total_df = total_df[keep_cols].copy()
    
    # Clean year columns (convert to numeric)
    for col in year_cols:
        total_df[col] = clean_numeric(total_df[col])
    
    # Melt to long format
    long_df = total_df.melt(
        id_vars=['itl_code'],
        var_name='year',
        value_name='value'
    )
    
    # Clean year column (handle '2023.0' format)
    long_df['year'] = pd.to_numeric(long_df['year'].astype(str).str.split('.').str[0], errors='coerce')
    
    # Add metric name
    long_df['metric'] = measure_name
    
    # Filter out UK total (we only want ITL1 regions)
    long_df = long_df[long_df['itl_code'] != 'UK'].copy()
    
    # Map ITL codes to region names
    long_df['region'] = long_df['itl_code'].map(ITL1_REGIONS)
    long_df = long_df.rename(columns={'itl_code': 'region_code'})
    
    # Drop nulls and duplicates
    long_df = long_df.dropna(subset=['year', 'value', 'region'])
    long_df = long_df[['region', 'region_code', 'year', 'metric', 'value']]
    
    return long_df

def main():
    logger.info(f"Processing GVA file: {IN}")
    
    # Load Excel file
    excel_file = pd.ExcelFile(IN)
    logger.info(f"Found {len(excel_file.sheet_names)} sheets")
    
    # Based on diagnostic output:
    # Table 1b: Chained volume in 2022 money value (£ million) - WANT THIS
    # Table 1c: Current price estimates (£ million) - WANT THIS
    
    all_data = []
    
    # Process Table 1b - Chained volume measures in pounds million
    if 'Table 1b' in excel_file.sheet_names:
        logger.info(f"Processing chained GVA from: Table 1b")
        df_chained = pd.read_excel(IN, sheet_name='Table 1b', header=None)
        chained_data = extract_itl1_data(df_chained, 'chained_gva_mn_gbp', is_index=False)
        if not chained_data.empty:
            all_data.append(chained_data)
            logger.info(f"  Extracted {len(chained_data)} chained GVA rows")
        else:
            logger.warning("No chained GVA data extracted from Table 1b")
    
    # Process Table 1c - Current price (nominal) in pounds million
    if 'Table 1c' in excel_file.sheet_names:
        logger.info(f"Processing nominal GVA from: Table 1c")
        df_nominal = pd.read_excel(IN, sheet_name='Table 1c', header=None)
        nominal_data = extract_itl1_data(df_nominal, 'nominal_gva_mn_gbp', is_index=False)
        if not nominal_data.empty:
            all_data.append(nominal_data)
            logger.info(f"  Extracted {len(nominal_data)} nominal GVA rows")
        else:
            logger.warning("No nominal GVA data extracted from Table 1c")
    
    if not all_data:
        raise ValueError("No data extracted from any sheets")
    
    # Combine all data
    long_df = pd.concat(all_data, ignore_index=True)
    
    # Sort and save long format
    long_df = long_df.sort_values(['region_code', 'metric', 'year']).reset_index(drop=True)
    long_df.to_csv(OUT_LONG, index=False)
    logger.info(f"Saved long format: {OUT_LONG}")
    
    # Create wide format
    wide_df = long_df.pivot_table(
        index=['region', 'region_code', 'metric'],
        columns='year',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    # Clean column names and sort
    wide_df.columns.name = None
    id_cols = ['region', 'region_code', 'metric']
    year_cols = sorted([c for c in wide_df.columns if c not in id_cols])
    wide_df = wide_df[id_cols + year_cols].sort_values(['region_code', 'metric']).reset_index(drop=True)
    
    wide_df.to_csv(OUT_WIDE, index=False)
    logger.info(f"Saved wide format: {OUT_WIDE}")
    
    # QA summary
    print("\n✅ GVA Cleaning Complete!")
    print(f"📊 Metrics: {sorted(long_df['metric'].unique())}")
    print(f"📅 Year span: {long_df['year'].min():.0f} → {long_df['year'].max():.0f}")
    print(f"🌍 Regions: {long_df['region_code'].nunique()} ITL1 regions")
    print(f"📝 Total rows (long): {len(long_df)}")
    print(f"📝 Total rows (wide): {len(wide_df)}")
    
    # Check for missing data
    if year_cols:
        null_count = pd.isna(wide_df[year_cols]).sum().sum()
        print(f"⚠️  Nulls in wide format: {null_count}")
    
    # Sample output
    print("\n📋 Sample output (first 3 rows of wide format):")
    display_cols = id_cols + year_cols[-3:] if len(year_cols) >= 3 else id_cols + year_cols
    print(wide_df[display_cols].head(3))

if __name__ == "__main__":
    main()