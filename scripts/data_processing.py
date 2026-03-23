import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# git pull
# git status
# git add
# git commit -m "message"
# git push


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# Margarita: [1] Merge Data, [2] Check Data types
# Svenja: [3] check and handle missing values
# Michael: [4] outlier check
# All three: [5] visualize -> think research questions...

# Load and pre-process GDP data
df_gdp = pd.read_csv('../data/processed/canton_gdp_2022_clean.csv')
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
df_listings = pd.read_csv('../data/processed/immobilier_all_cantons_allpages_snapshot_p20.csv')

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

# Check missing values
print("Missing values before cleaning:")
print(df.isna().sum())
# No missing values found in key variables (canton, price_chf, canton_gdp)
# Therefore, no row deletion or merge correction was necessary

# Handle missing values
for col in ['living_area_m2', 'rooms']:
    df[col] = df.groupby('canton')[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())

print(df.isna().sum())
# Missing values in living_area_m2 and rooms were imputed using the median per canton to account for regional differences.
# Remaining missing values were filled using the overall median.
# Missing values in location_text were not treated, as this variable is not relevant for the analysis.


### OUTlIER ANALYSIS ###
# Variables to check
cols = ['price_chf', 'living_area_m2', 'rooms']

# Summary statistics
print(df[cols].describe())

# Boxplots
plt.figure(figsize=(12, 6))

for i, col in enumerate(cols, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()

"""
# price_chf: median = 1590, mean = 2119, max = 230000 → strong right skew, extreme outliers
# living_area_m2: median = 81.5, mean = 89.5, max = 800 → moderate skew, large property outliers
# rooms: median = 3.5, mean = 3.59, max = 14 → relatively stable, few large values
# mean > median → indicates skewness (especially for price_chf)
"""

# Outliers were not removed because they represent real market observations,
# particularly luxury properties rather than data errors.

# Removing these values could introduce bias and distort the true distribution
# of rental prices across cantons.

# Instead of removing outliers, a log transformation was applied to reduce skewness
# and improve the robustness of the analysis.


# 1. Create price per square meter
# This feature normalizes rental price by property size,
# making comparisons across different properties more meaningful.
df['price_per_m2'] = df['price_chf'] / df['living_area_m2']


# 2. Apply log transformation to price
# Rental prices are highly right-skewed due to extreme high-value properties.
# Log transformation reduces skewness and stabilizes variance,
# making the distribution more suitable for analysis and modeling.
df['log_price'] = np.log(df['price_chf'])
df['log_price_per_m2'] = np.log(df['price_per_m2'])

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(df.head())


cols = ['price_chf', 'log_price', 'price_per_m2', 'log_price_per_m2']

plt.figure(figsize=(14, 4))

for i, col in enumerate(cols, 1):
    plt.subplot(1, 4, i)
    sns.boxplot(y=df[col])
    plt.title(col)
    plt.ylabel(col)

plt.tight_layout()
plt.show()

""" Outliers were retained as they represent genuine high-value properties
rather than data errors. Removing them could bias the analysis by excluding 
important segments of the housing market. Instead, log transformation was
applied to reduce their influence."""



### Scatter plot ###

plt.figure(figsize=(6,4))
sns.scatterplot(x=df['canton_gdp'], y=df['log_price_per_m2'])

plt.xlabel("Canton GDP")
plt.ylabel("Log Price per m²")
plt.title("GDP vs Log Price per m²")
plt.show()

# The scatter plot suggests a weak positive relationship between canton GDP
# and rental prices per square meter. While higher GDP cantons tend to have
# slightly higher rental prices, the large spread within each canton indicates
# that GDP alone is not a strong predictor of rental prices.

### Correlation ###

corr = df[['canton_gdp', 'log_price_per_m2']].corr()
print(corr)

#                   canton_gdp  log_price_per_m2
# canton_gdp          1.000000          0.199271
# log_price_per_m2    0.199271          1.000000

# The weak relationship suggests that other factors, such as location-specific demand,
# urbanization, and housing market dynamics, play a more significant role than GDP alone.


### Correlation Heatmap ###
plt.figure(figsize=(8,6))
sns.heatmap(df[['price_chf','price_per_m2','log_price','log_price_per_m2','canton_gdp']].corr(),
            annot=True, cmap='coolwarm', fmt=".2f")

plt.title("Correlation Matrix")
plt.show()

"""The relationship between price_chf and price_per_m2 is moderate (≈ 0.53).
The strongest relationship is observed between log_price and log_price_per_m2 (≈ 0.58).
This indicates that the transformations are consistent and meaningful.
The relationship between canton_gdp and other variables is very weak (0.03 – 0.20).
The highest correlation is between GDP and log_price_per_m2 (≈ 0.20), which is still weak"""


### Canton Based ###
plt.figure(figsize=(12,5))
sns.boxplot(x='canton', y='log_price_per_m2', data=df)
plt.xticks(rotation=90)
plt.title("Log Price per m² by Canton")
plt.show()

# Outliers?

### Median Rent vs GDP ###
df_grouped = df.groupby('canton').agg({
    'price_per_m2': 'median',
    'canton_gdp': 'first'
}).reset_index()

sns.scatterplot(x='canton_gdp', y='price_per_m2', data=df_grouped)

plt.title("Median Price per m² vs GDP (Canton Level)")
plt.show()