#!/usr/bin/env python3
"""
Diagnostic script to understand the structure of the ONS GVA Excel file.
This will help us fix the cleaning script.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Find the GVA file
IN_DIR = Path("data/raw/gva")
gva_files = sorted(IN_DIR.glob("gva_*.xlsx"))
if not gva_files:
    raise FileNotFoundError(f"No GVA files found in {IN_DIR}")
IN = gva_files[-1]

print(f"Analyzing file: {IN}")
print("=" * 80)

# Load Excel file
excel_file = pd.ExcelFile(IN)
print(f"\nFound {len(excel_file.sheet_names)} sheets:")
for i, sheet in enumerate(excel_file.sheet_names):
    print(f"  {i}: {sheet}")

print("\n" + "=" * 80)
print("ANALYZING TABLE 1a (Nominal GVA)")
print("=" * 80)

# Read Table 1a without any headers to see raw structure
df_1a = pd.read_excel(IN, sheet_name='Table 1a', header=None)
print(f"\nShape: {df_1a.shape}")
print("\nFirst 20 rows, first 10 columns:")
print(df_1a.iloc[:20, :10])

print("\n" + "-" * 40)
print("Column values in row 5-15 (looking for headers):")
for i in range(5, min(15, len(df_1a))):
    row_vals = df_1a.iloc[i].dropna().astype(str).tolist()
    if row_vals:
        print(f"Row {i}: {row_vals[:10]}...")  # First 10 non-null values

print("\n" + "=" * 80)
print("ANALYZING TABLE 1b (Chained GVA)")
print("=" * 80)

df_1b = pd.read_excel(IN, sheet_name='Table 1b', header=None)
print(f"\nShape: {df_1b.shape}")
print("\nFirst 20 rows, first 10 columns:")
print(df_1b.iloc[:20, :10])

print("\n" + "-" * 40)
print("Looking for year columns in different rows...")

# Check multiple potential header rows
for sheet_name in ['Table 1a', 'Table 1b']:
    print(f"\n{sheet_name} - Searching for years:")
    df = pd.read_excel(IN, sheet_name=sheet_name, header=None)
    
    for row_idx in range(min(20, len(df))):
        row = df.iloc[row_idx]
        # Look for 4-digit years
        potential_years = []
        for val in row:
            str_val = str(val).strip()
            if str_val.isdigit() and len(str_val) == 4 and 1990 <= int(str_val) <= 2030:
                potential_years.append(str_val)
        
        if potential_years:
            print(f"  Row {row_idx}: Found years: {potential_years[:5]}...")
            break

print("\n" + "=" * 80)
print("CHECKING FOR ITL CODES")
print("=" * 80)

# Look for ITL codes
for sheet_name in ['Table 1a', 'Table 1b']:
    print(f"\n{sheet_name}:")
    df = pd.read_excel(IN, sheet_name=sheet_name, header=None)
    
    # Search for TL codes in first few columns
    for col_idx in range(min(5, df.shape[1])):
        col = df.iloc[:, col_idx].astype(str)
        tl_codes = col[col.str.startswith('TL', na=False)]
        if not tl_codes.empty:
            print(f"  Column {col_idx} contains ITL codes:")
            print(f"    First few: {tl_codes.head(10).tolist()}")
            print(f"    Unique count: {tl_codes.nunique()}")
            break

print("\n" + "=" * 80)
print("SAMPLE DATA EXTRACTION")
print("=" * 80)

# Try to read with pandas' auto-detection
for sheet_name in ['Table 1a', 'Table 1b']:
    print(f"\n{sheet_name} - Using pandas auto-header detection:")
    try:
        df_auto = pd.read_excel(IN, sheet_name=sheet_name)
        print(f"  Shape: {df_auto.shape}")
        print(f"  Columns: {list(df_auto.columns[:10])}...")
        print(f"  First few rows:")
        print(df_auto.head(3))
    except Exception as e:
        print(f"  Error: {e}")