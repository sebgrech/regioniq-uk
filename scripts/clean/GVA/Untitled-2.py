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

def find_header_row(df, keywords=['ITL', 'Region', 'SIC']):
    """Find the row containing column headers."""
    for idx, row in df.iterrows():
        row_str = ' '.join(row.astype(str).fillna(''))
        if any(kw in row_str for kw in keywords):
            return idx
    return 0

def extract_itl1_data(sheet_df, measure_name):
    """
    Extract ITL1 total GVA from a Table 1 sheet.
    
    Args:
        sheet_df: Raw dataframe from Excel sheet
        measure_name: 'nominal_gva_mn_gbp' or 'chained_gva_mn_gbp'
    
    Returns:
        DataFrame with columns: region, region_code, year, metric, value
    """
    # Find header row
    header_idx = find_header_row(sheet_df)
    
    # Use header row as column names
    sheet_df.columns = sheet_df.iloc[header_idx].fillna('').astype(str)
    sheet_df = sheet_df.iloc[header_idx + 1:].reset_index(drop=True)
    
    # Rename first few columns consistently
    col_mapping = {}
    for i, col in enumerate(sheet_df.columns[:5]):
        if 'ITL' in col or 'code' in col.lower():
            col_mapping[col] = 'itl_code'
        elif 'region' in col.lower() or 'name' in col.lower():
            col_mapping[col] = 'region_name'
        elif 'SIC' in col or 'industry' in col.lower():
            col_mapping[col] = 'industry'
    
    sheet_df = sheet_df.rename(columns=col_mapping)
    
    # Identify year columns (4-digit numbers)
    year_cols = []
    for col in sheet_df.columns:
        if re.match(r'^\d{4}$', str(col)):
            year_cols.append(col)
    
    if not year_cols:
        logger.warning(f"No year columns found for {measure_name}")
        return pd.DataFrame()
    
    # Filter for ITL1 regions and total GVA
    itl1_df = sheet_df[sheet_df['itl_code'].isin(ITL1_REGIONS.keys())].copy()
    
    # Get total rows (usually industry code 'A-T' or contains 'Total')
    if 'industry' in itl1_df.columns:
        total_df = itl1_df[
            (itl1_df['industry'].str.contains('Total', case=False, na=False)) |
            (itl1_df['industry'] == 'A-T') |
            (itl1_df['industry'].str.contains('All industries', case=False, na=False))
        ]
    else:
        # If no industry column, assume all rows are totals
        total_df = itl1_df
    
    if total_df.empty:
        logger.warning(f"No total GVA rows found for {measure_name}")
        return pd.DataFrame()
    
    # Keep only relevant columns
    keep_cols = ['itl_code'] + year_cols
    total_df = total_df[keep_cols].copy()
    
    # Melt to long format
    long_df = total_df.melt(
        id_vars=['itl_code'],
        var_name='year',
        value_name='value'
    )
    
    # Clean and add metadata
    long_df['year'] = pd.to_numeric(long_df['year'], errors='coerce')
    long_df['value'] = clean_numeric(long_df['value'])
    long_df['metric'] = measure_name
    
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
    
    # Find Table 1 sheets (nominal and chained)
    nominal_sheet = None
    chained_sheet = None
    
    for sheet in excel_file.sheet_names:
        sheet_lower = sheet.lower()
        if 'table 1' in sheet_lower:
            if 'current' in sheet_lower or 'nominal' in sheet_lower:
                nominal_sheet = sheet
            elif 'chained' in sheet_lower or 'cvm' in sheet_lower or 'volume' in sheet_lower:
                chained_sheet = sheet
    
    if not nominal_sheet and not chained_sheet:
        # Fallback: look for sheets with just "Table 1" 
        table1_sheets = [s for s in excel_file.sheet_names if 'Table 1' in s]
        if len(table1_sheets) >= 2:
            # Assume first is nominal, second is chained (common pattern)
            nominal_sheet = table1_sheets[0]
            chained_sheet = table1_sheets[1]
    
    logger.info(f"Nominal sheet: {nominal_sheet}")
    logger.info(f"Chained sheet: {chained_sheet}")
    
    # Process each measure
    all_data = []
    
    if nominal_sheet:
        logger.info(f"Processing nominal GVA from: {nominal_sheet}")
        df_nominal = pd.read_excel(IN, sheet_name=nominal_sheet, header=None)
        nominal_data = extract_itl1_data(df_nominal, 'nominal_gva_mn_gbp')
        if not nominal_data.empty:
            all_data.append(nominal_data)
            logger.info(f"  Extracted {len(nominal_data)} nominal GVA rows")
    
    if chained_sheet:
        logger.info(f"Processing chained GVA from: {chained_sheet}")
        df_chained = pd.read_excel(IN, sheet_name=chained_sheet, header=None)
        chained_data = extract_itl1_data(df_chained, 'chained_gva_mn_gbp')
        if not chained_data.empty:
            all_data.append(chained_data)
            logger.info(f"  Extracted {len(chained_data)} chained GVA rows")
    
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
    print(wide_df[id_cols + year_cols[-3:]].head(3))

if __name__ == "__main__":
    main()