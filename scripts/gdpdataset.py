import pandas as pd
from pathlib import Path

# paths
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]  # -> .../project
input_path = BASE_DIR / "data" / "raw" / "gdp_percapita_canton_2022.xlsx"
output_path = BASE_DIR / "data" / "processed" / "canton_gdp_2022_clean.csv"

# read excel
df = pd.read_excel(input_path)

print("Original columns:", df.columns)

# select Canton + 2022 (2022 INT column!)
df_selected = df[["Canton", 2022]].copy()

# clean Canton names (remove ":" and spaces)
df_selected["Canton"] = (
    df_selected["Canton"]
    .astype(str)
    .str.replace(":", "", regex=False)
    .str.strip()
)

# round GDP
df_selected[2022] = df_selected[2022].round(2)

# save csv
df_selected.to_csv(output_path, index=False)
print(df_selected.head())