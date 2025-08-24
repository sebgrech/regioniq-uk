#!/usr/bin/env python3
"""
Debug the column structure to understand why we're getting a DataFrame instead of Series.
"""

from pathlib import Path
import pandas as pd

# Find the GVA file
IN_DIR = Path("data/raw/gva")
gva_files = sorted(IN_DIR.glob("gva_*.xlsx"))
if not gva_files:
    raise FileNotFoundError(f"No GVA files found in {IN_DIR}")
IN = gva_files[-1]

print(f"Debugging column structure in: {IN}")
print("=" * 80)

# Check Table 1b structure
sheet_name = 'Table 1b'
print(f"\nAnalyzing {sheet_name}:")
print("-" * 40)

# Read the sheet
df = pd.read_excel(IN, sheet_name=sheet_name, header=None)

# Show what's in row 1 (the header row)
print("\nRow 1 (header row) values:")
header_row = df.iloc[1]
for i, val in enumerate(header_row[:10]):
    print(f"  Column {i}: '{val}'")

# Set headers from row 1
df.columns = df.iloc[1].fillna('').astype(str)
df = df.iloc[2:].reset_index(drop=True)

# Clean column names
df.columns = [str(col).strip() for col in df.columns]

print("\nColumn names after setting from row 1:")
for i, col in enumerate(df.columns[:10]):
    print(f"  {i}: '{col}'")

# Check for duplicate column names
print("\nChecking for duplicate column names:")
from collections import Counter
col_counts = Counter(df.columns)
duplicates = {col: count for col, count in col_counts.items() if count > 1}
if duplicates:
    print("  Found duplicates:")
    for col, count in duplicates.items():
        print(f"    '{col}': appears {count} times")
else:
    print("  No duplicates found")

# Show the type of what we get when accessing columns
print("\nColumn access diagnostics:")
print(f"  type(df['ITL code']): {type(df.get('ITL code', 'Not found'))}")
print(f"  type(df['SIC07 code']): {type(df.get('SIC07 code', 'Not found'))}")

# Try the renaming logic
col_mapping = {}
for i, col in enumerate(df.columns[:4]):
    print(f"\nColumn {i}: '{col}'")
    if 'ITL' in col or i == 0:
        col_mapping[col] = 'itl_code'
        print(f"  -> Mapping to 'itl_code'")
    elif 'Region' in col or i == 1:
        col_mapping[col] = 'region_name'
        print(f"  -> Mapping to 'region_name'")
    elif 'SIC' in col or i == 2:
        col_mapping[col] = 'sic_code'
        print(f"  -> Mapping to 'sic_code'")
    elif 'Industry' in col or i == 3:
        col_mapping[col] = 'industry'
        print(f"  -> Mapping to 'industry'")

print(f"\nFinal mapping: {col_mapping}")

# Apply the mapping
df_renamed = df.rename(columns=col_mapping)
print(f"\nColumns after renaming (first 10):")
for i, col in enumerate(df_renamed.columns[:10]):
    print(f"  {i}: '{col}'")

# Check what happens when we access sic_code
if 'sic_code' in df_renamed.columns:
    sic_col = df_renamed['sic_code']
    print(f"\ntype(df_renamed['sic_code']): {type(sic_col)}")
    if isinstance(sic_col, pd.DataFrame):
        print("  ERROR: It's a DataFrame! This means there are duplicate 'sic_code' columns")
        print(f"  Shape: {sic_col.shape}")
    else:
        print(f"  Good: It's a Series with shape {sic_col.shape}")
        print(f"  First few values: {sic_col.head().tolist()}")