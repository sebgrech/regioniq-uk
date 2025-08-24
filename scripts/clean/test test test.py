import pandas as pd
from pathlib import Path

IN = Path("data/raw/population_LAD_nomis.csv")
df = pd.read_csv(IN)

print("SEX_NAME uniques →")
print(df["SEX_NAME"].unique())
