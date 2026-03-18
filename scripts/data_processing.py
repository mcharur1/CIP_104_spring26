import pandas as pd

# THIS IS THE NEW SCRIPT

a = "some varibale"

# Margarita: [1] Merge Data, [2] Check Data types
# Svenja: [3] check and handle missing values
# Michael: [4] outlier check
# All three: [5] visualize -> think research questions...

# Load and pre-process GDP data
df_gdp = pd.read_csv('data/processed/immobilier_all_cantons_allpages_snapshot_p20.csv')
df_gdp = df_gdp.rename(columns={'Canton': 'canton_name'})

# Add abbreviated canton column to GDP for later join
CANTONS = {
    "ZH": "zurich",
    "BE": "berne",
    "LU": "lucerne",
    "UR": "uri",
    "SZ": "schwyz",
    "OW": "obwald",
    "NW": "nidwald",
    "GL": "glaris",
    "ZG": "zoug",
    "FR": "fribourg",
    "SO": "soleure",
    "BS": "bale-ville",
    "BL": "bale-campagne",
    "SH": "schaffhouse",
    "AR": "appenzell-rhodes-exterieures",
    "AI": "appenzell-rhodes-interieures",
    "SG": "st-gall",
    "GR": "grisons",
    "AG": "argovie",
    "TG": "thurgovie",
    "TI": "tessin",
    "VD": "vaud",
    "VS": "valais",
    "NE": "neuchatel",
    "GE": "geneva",
    "JU": "jura",
}
print(df_gdp['Canton'].unique())

df = pd.merge(df_listings, df_gdp, on'Canton', how='left')

# Load and pre-process listing data
df_listings = pd.read_csv('/Users/mcharur1/PycharmProjects/CIP/W4/immobilier_all_cantons_allpages_snapshot_p20.csv')

# Merge GDP and Listings data