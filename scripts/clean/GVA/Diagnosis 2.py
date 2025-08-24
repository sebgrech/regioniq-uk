#!/usr/bin/env python3
"""
Quick check to identify which sheet contains nominal (current prices) GVA data.
"""

from pathlib import Path
import pandas as pd

# Find the GVA file
IN_DIR = Path("data/raw/gva")
gva_files = sorted(IN_DIR.glob("gva_*.xlsx"))
if not gva_files:
    raise FileNotFoundError(f"No GVA files found in {IN_DIR}")
IN = gva_files[-1]

print(f"Checking sheets in: {IN}")
print("=" * 80)

# Check each Table 1 sheet
for sheet_name in ['Table 1a', 'Table 1b', 'Table 1c', 'Table 1d']:
    print(f"\n{sheet_name}:")
    print("-" * 40)
    
    # Read the title row
    df = pd.read_excel(IN, sheet_name=sheet_name, nrows=2, header=None)
    
    # Print the title (first row, first column)
    title = df.iloc[0, 0] if not df.empty else "No title"
    print(f"Title: {title}")
    
    # Check for key indicators
    title_lower = str(title).lower()
    if 'current' in title_lower:
        print("  ✓ Contains 'current' - likely NOMINAL prices")
    if 'chained' in title_lower:
        print("  ✓ Contains 'chained' - likely REAL/VOLUME measures")
    if 'index' in title_lower:
        print("  ✓ Contains 'index' - INDEX format (not money values)")
    if 'million' in title_lower or 'money value' in title_lower:
        print("  ✓ Contains 'million/money value' - POUNDS MILLION format")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("- For nominal GVA in £ million: Use the sheet with 'current' and 'million/money value'")
print("- For chained GVA in £ million: Use Table 1b (chained volume in 2022 money value)")