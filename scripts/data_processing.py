import pandas as pd

# git pull
# git status
# git add
# git commit -m "message"
# git push

# Margarita: [1] Merge Data, [2] Check Data types
# Svenja: [3] check and handle missing values
# Michael: [4] outlier check
# All three: [5] visualize -> think research questions...

# Load and pre-process GDP data
df_gdp = pd.read_csv('data/processed/canton_gdp_2022_clean.csv')
df_gdp = df_gdp.rename(columns={'Canton': 'canton_name', '2022': 'canton_gdp'})

# Add abbreviated canton column to GDP for later join
CANTONS = {
    "ZH": "Zurich",
    "BE": "Berne",
    "LU": "Lucerne",
    "UR": "Uri",
    "SZ": "Schwyz",
    "OW": "Obwalden",
    "NW": "Nidwalden",
    "GL": "Glarus",
    "ZG": "Zug",
    "FR": "Fribourg",
    "SO": "Solothurn",
    "BS": "Basel-Stadt",
    "BL": "Basel-Landschaft",
    "SH": "Schaffhausen",
    "AR": "Appenzell A. Rh.",
    "AI": "Appenzell I. Rh.",
    "SG": "St. Gallen",
    "GR": "Graubünden",
    "AG": "Aargau",
    "TG": "Thurgau",
    "TI": "Ticino",
    "VD": "Vaud",
    "VS": "Valais",
    "NE": "Neuchâtel",
    "GE": "Geneva",
    "JU": "Jura",
    "CH": "Switzerland"
}

df_cantons = pd.DataFrame(list(CANTONS.items()), columns=['canton', 'canton_name'])
df_gdp = pd.merge(df_gdp, df_cantons, on='canton_name', how='left')

# Load and pre-process listing data
df_listings = pd.read_csv('data/processed/immobilier_all_cantons_allpages_snapshot_p20.csv')

# Merge Listing and GDP data
df = pd.merge(df_listings, df_gdp, on='canton', how='left')

# Changing date to date type (previously str)
df['date_scraped'] = pd.to_datetime(df['date_scraped'])

# Data types
print(df.dtypes) # all date types are correct for the variable they represent.
# canton                       str
# date_scraped      datetime64[us]
# price_chf                float64
# living_area_m2           float64
# rooms                    float64
# location_text                str
# listing_url                  str
# source                       str
# canton_name                  str
# canton_gdp               float64


