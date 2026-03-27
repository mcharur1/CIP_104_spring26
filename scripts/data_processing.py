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


### Margarita's plots
'''
In order to look at price impacts three main drivers were divided into tiers so we could see better how prices behave across different groups 
for each driver. Then a boxplot is created for each driver to show the spread of the data and whether it changes by groups within each driver.
Some findings are:
1. GDP tier shows that high GDP per capita does not clearly separate the market showing overlapping charts for all groups.
2. The room actually shows very similar distribution for the highest and lowest groups. This could make sense if we think that the initial cost of 
an apartment is very similar and each added room only adds a marginal cost if none at all per m2.
3. The living would be the most clear difference between the smallest and the largest group, where interestingly enough the smaller the apartment the 
more expensive. This could be due to the fact that more expensive areas typically have smaller living spaces due to demand yet, the prices are very 
high because of the high-demand location.

All in all, although not logically what I expected, it seems that the most clear driver for price changes is living area.
'''
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

df['gdp_tiers'] = pd.cut(
    df['canton_gdp'],
    bins=[0, 80_000, 110_000, float('inf')],
    labels=['Low GDP\n(<80k)', 'Medium GDP\n(80–110k)', 'High GDP\n(>110k)'])
ORDER = ['Low GDP\n(<80k)', 'Medium GDP\n(80–110k)', 'High GDP\n(>110k)']

# Price by GDP tier — the main question
sns.boxplot(x='gdp_tiers', y='log_price_per_m2', data=df,
            order=ORDER, hue='gdp_tiers', legend=False, ax=axes[0, 0])
axes[0, 0].set_title("Log price per m² by GDP tier", fontsize=12)
axes[0, 0].set_xlabel("GDP tier")
axes[0, 0].set_ylabel("Log price per m² (ln CHF/m²)")

# Price by room count — bin rooms first so it's readable
df['rooms_tiers'] = pd.cut(df['rooms'], bins=[0,2,3,4,5,float('inf')],
                             labels=['≤2','3','4','5','6+'])
sns.boxplot(x='rooms_tiers', y='log_price_per_m2', data=df,
            hue='rooms_tiers', legend=False, ax=axes[0, 1])
axes[0, 1].set_title("Log price per m² by room count", fontsize=12)
axes[0, 1].set_xlabel("Rooms")
axes[0, 1].set_ylabel("")

# Price by living area — bin area into readable brackets
df['area_tiers'] = pd.cut(df['living_area_m2'], bins=[0,50,80,120,float('inf')],
                            labels=['<50m²','50–80m²','80–120m²','>120m²'])
sns.boxplot(x='area_tiers', y='log_price_per_m2', data=df,
            hue='area_tiers', legend=False, ax=axes[1, 0])
axes[1, 0].set_title("Log price per m² by living area", fontsize=12)
axes[1, 0].set_xlabel("Living area")
axes[1, 0].set_ylabel("")

axes[1, 1].set_visible(False)

plt.suptitle("What drives log price per m²?", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()


### Svenja's plots
"""
The same GDP tier classification was used as before to ensure consistency across results.
To avoid redundancy, the GDP tiers should ideally be created once during the data preparation and reused throughout the analysis
"""

"""
Apartments in high-GDP cantons tend to be slightly larger on average. However, the distributions overlap across all GDP tiers,
so apartments in strong and weak cantons are often similar in size, meaning there are no clear differences based solely on GDP.
Regarding the heatmap: On average, apartments in high-GDP cantons are larger and have slightly more rooms. However, these differences are moderate.
A negative relationship can be observed between living area and price per square meter,
indicating that smaller apartments tend to be more expensive per m² than larger ones (maybe because of typical housing market dynamics,
where fixed costs and higher demand for smaller units lead to higher prices per square meter?).
This suggests that housing characteristics, particularly living area, help explain differences in rental prices beyond GDP per capita.
"""

# Is there a difference between strong and weak cantons?
plt.figure(figsize=(8, 5))
sns.boxplot(x='gdp_tiers', y='living_area_m2', data=df)
plt.title('Living area by GDP tier')
plt.xlabel('GDP tier')
plt.ylabel('Living area (m²)')
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(x='gdp_tiers', y='living_area_m2', data=df)
plt.title('Distribution of living area by GDP tier')
plt.xlabel('GDP tier')
plt.ylabel('Living area (m²)')
plt.show()

# Heatmap
grouped = df.groupby('gdp_tiers', observed=False)[['living_area_m2', 'rooms']].mean()
plt.figure(figsize=(6, 4))
sns.heatmap(grouped, annot=True, cmap='coolwarm')
plt.title('Average housing characteristics by GDP tier')
plt.show()

# Housing Characteristics vs. Price (with GDP as color)
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x='living_area_m2',
    y='price_per_m2',
    hue='gdp_tiers',
    data=df
)
plt.title('Price per m² vs living area')
plt.xlabel('Living area (m²)')
plt.ylabel('Price per m²')
plt.show()

# Does living area explain prices independently of GDP?
sns.lmplot(
    x='living_area_m2',
    y='price_per_m2',
    hue='gdp_tiers',
    data=df,
    height=5,
    aspect=1.4
)
plt.title('Relationship between living area and price per m² by GDP tier')
plt.show()

#  Michael Setup
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#  Colour palette
TIER_COLORS = {
    'Low\n(<80k)': '#7BAFD4',
    'Medium\n(80–110k)': '#F4A261',
    'High\n(>110k)': '#E76F51',
}
palette = list(TIER_COLORS.values())

df['price_per_m2'] = df['price_chf'] / df['living_area_m2']
df['log_price_per_m2'] = np.log(df['price_per_m2'])

tier_labels = ['Low\n(<80k)', 'Medium\n(80–110k)', 'High\n(>110k)']
df['gdp_tier'] = pd.cut(
    df['canton_gdp'],
    bins=[0, 80_000, 110_000, float('inf')],
    labels=tier_labels,
)

# Canton-level aggregates (used in plot 3)
canton_med = (
    df.groupby(['canton', 'gdp_tier'], observed=True)
    .agg(
        median_area=('living_area_m2', 'median'),
        median_rooms=('rooms', 'median'),
        median_price_m2=('price_per_m2', 'median'),
        canton_gdp=('canton_gdp', 'first'),
    )
    .reset_index()
)

fig = plt.figure(figsize=(16, 13))
fig.patch.set_facecolor('#F7F7F7')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

TITLE_KW = dict(fontsize=13, fontweight='bold', pad=10, color='#222222')
LABEL_KW = dict(fontsize=11, color='#444444')
TICK_KW = dict(labelsize=9)
ax1 = fig.add_subplot(gs[0, 0])

char_stats = (
    df.groupby('gdp_tier', observed=True)
    .agg(median_area=('living_area_m2', 'median'),
         median_rooms=('rooms', 'median'))
    .reindex(tier_labels)
)

x, w = np.arange(len(tier_labels)), 0.35
b1 = ax1.bar(x - w / 2, char_stats['median_area'], width=w,
             color=palette, edgecolor='white', linewidth=0.8)
b2 = ax1.bar(x + w / 2, char_stats['median_rooms'] * 20, width=w,
             color=palette, edgecolor='white', linewidth=0.8, alpha=0.55, hatch='//')

for bar, val in zip(b1, char_stats['median_area']):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{val:.0f} m²', ha='center', va='bottom', fontsize=9, color='#333333')
for bar, val in zip(b2, char_stats['median_rooms']):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{val:.1f} rm', ha='center', va='bottom', fontsize=9, color='#555555')

ax1.set_xticks(x)
ax1.set_xticklabels(tier_labels, fontsize=9)
ax1.set_ylabel('Median living area (m²)', **LABEL_KW)
ax1.set_title('Housing Characteristics by GDP Tier', **TITLE_KW)
ax1.set_facecolor('#FAFAFA')
ax1.spines[['top', 'right']].set_visible(False)

solid_p = mpatches.Patch(facecolor='#aaa', label='Median area (m²)')
hatch_p = mpatches.Patch(facecolor='#aaa', hatch='//', alpha=0.55, label='Median rooms (×20)')
ax1.legend(handles=[solid_p, hatch_p], fontsize=9)

ax3 = fig.add_subplot(gs[1, 0])

for tier, color in TIER_COLORS.items():
    sub = canton_med[canton_med['gdp_tier'] == tier]
    ax3.scatter(sub['median_area'], sub['median_price_m2'],
                c=color, s=90, edgecolors='white', linewidths=0.8,
                label=tier.replace('\n', ' '), zorder=3, alpha=0.9)
    if len(sub) > 2:
        m, b, *_ = stats.linregress(sub['median_area'], sub['median_price_m2'])
        xs = np.linspace(sub['median_area'].min(), sub['median_area'].max(), 50)
        ax3.plot(xs, m * xs + b, color=color, linewidth=1.5, alpha=0.6, linestyle='--')

notable = ['ZG', 'GE', 'BS', 'UR', 'JU']
for _, row in canton_med[canton_med['canton'].isin(notable)].iterrows():
    ax3.annotate(row['canton'], (row['median_area'], row['median_price_m2']),
                 textcoords='offset points', xytext=(5, 3), fontsize=8, color='#333333')

ax3.set_xlabel('Median Living Area (m²)', **LABEL_KW)
ax3.set_ylabel('Median Price per m² (CHF)', **LABEL_KW)
ax3.set_title('Size vs. Price per m² at Canton Level', **TITLE_KW)
ax3.set_facecolor('#FAFAFA')
ax3.spines[['top', 'right']].set_visible(False)
ax3.legend(title='GDP tier', fontsize=9, title_fontsize=9)
ax3.tick_params(**TICK_KW)

## Plot 3 – Partial R² bar
ax4 = fig.add_subplot(gs[1, 1])

df_m = df[['log_price_per_m2', 'canton_gdp', 'living_area_m2', 'rooms']].dropna()
y = df_m['log_price_per_m2'].values


def partial_r2(full_r2, reduced_r2):
    return (full_r2 - reduced_r2) / (1 - reduced_r2)


def fit_r2(X):
    model = LinearRegression().fit(X, y)
    return r2_score(y, model.predict(X))


r2_full = fit_r2(df_m[['canton_gdp', 'living_area_m2', 'rooms']].values)
r2_no_gdp = fit_r2(df_m[['living_area_m2', 'rooms']].values)
r2_no_area = fit_r2(df_m[['canton_gdp', 'rooms']].values)
r2_no_rooms = fit_r2(df_m[['canton_gdp', 'living_area_m2']].values)

partial = {
    'Canton GDP': partial_r2(r2_full, r2_no_gdp),
    'Living Area': partial_r2(r2_full, r2_no_area),
    'Rooms': partial_r2(r2_full, r2_no_rooms),
}

bar_colors = ['#7BAFD4', '#E76F51', '#F4A261']
bars = ax4.barh(list(partial.keys()), list(partial.values()),
                color=bar_colors, edgecolor='white', height=0.5)
for bar, val in zip(bars, partial.values()):
    ax4.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
             f'{val:.3f}', va='center', fontsize=10, color='#333333')

ax4.set_xlabel('Partial R² (unique variance explained)', **LABEL_KW)
ax4.set_title('Unique Explanatory Power\nfor Log Price per m²', **TITLE_KW)
ax4.set_facecolor('#FAFAFA')
ax4.spines[['top', 'right']].set_visible(False)
ax4.tick_params(**TICK_KW)
ax4.set_xlim(0, max(partial.values()) * 1.3)
ax4.text(0.98, 0.05, f'Full model R² = {r2_full:.3f}',
         transform=ax4.transAxes, ha='right', fontsize=9,
         color='#666666', style='italic')



## Final plots
fig.suptitle(
    'Housing Characteristics across GDP Tiers & Their Role in Explaining Rental Prices',
    fontsize=14, fontweight='bold', color='#111111', y=1.01,
)

plt.show()

## Housing Characteristics by GDP Tier -
# Higher-GDP cantons do have larger apartments
# (median ~88 m² vs ~80 m² in low-GDP cantons)
# and a fractionally higher room count.
# The differences are real but modest — housing size does vary
# with economic strength, but not dramatically.

## Size vs. Price per m² at Canton Level -
# This is the most striking plot. Across all GDP tiers,
# there's a negative relationship between
# apartment size and price/m² — smaller apartments cost more
# per square meter. This holds within
# each tier (dashed regression lines), and notable cantons
# like Zug (ZG) and Geneva (GE) command
# premium prices despite not being the largest units.

## Partial R² (Unique Explanatory Power) -
# This quantifies exactly how much each variable uniquely
# explains in log price/m²,
# holding the others constant. Canton GDP (partial R² ≈ 0.042)
# actually explains more
# than living area (≈ 0.030) or rooms (≈ 0.017) individually —
# but all three together only
# explain ~7% of variance, meaning other factors (location,
# transport access, amenities) dominate.


### SEDA ###

import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Three regression models are built to compare how well each factor group
# explains log price per m². Each model isolates one factor group.
model_gdp = smf.ols('log_price_per_m2 ~ canton_gdp', data=df).fit()
model_housing = smf.ols('log_price_per_m2 ~ living_area_m2 + rooms', data=df).fit()
model_full = smf.ols('log_price_per_m2 ~ canton_gdp + living_area_m2 + rooms', data=df).fit()

# R² values from each model are extracted and visualised as a bar chart
# to directly compare the explanatory power of each factor group.
models = ['Economic\n(GDP)', 'Housing\nCharacteristics', 'Combined']
r2_values = [model_gdp.rsquared, model_housing.rsquared, model_full.rsquared]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, r2_values, color=['steelblue', 'coral', 'mediumseagreen'])
plt.ylabel("R² (Explained Variance)")
plt.title("Q3: Which factors explain rental prices per m²?")
plt.ylim(0, 1)

# R² values are displayed on top of each bar for readability
for bar, val in zip(bars, r2_values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f'{val:.3f}', ha='center', fontsize=11)

plt.tight_layout()
plt.show()

# A new feature 'city' is extracted from location_text by removing
# the 4-digit postal code prefix where present.
# For example: '8600 Dübendorf' → 'Dübendorf', 'Zürich' → 'Zürich' (unchanged)
# The postal code is only removed for cleaning purposes — it does not determine
# whether a listing is central or not. That decision is made separately
# using the canton_centers dictionary based on the city name.


df['city'] = df['location_text'].str.replace(r'^\d{4}\s*', '', regex=True).str.strip()
print(df['city'].value_counts().head(20))

# Each canton's administrative center (capital city) is defined manually
# to serve as a proxy for central urban location
canton_centers = {
    'ZH': 'Zürich',
    'BE': 'Bern',
    'LU': 'Luzern',
    'UR': 'Altdorf',
    'SZ': 'Schwyz',
    'OW': 'Sarnen',
    'NW': 'Stans',
    'GL': 'Glarus',
    'ZG': 'Zug',
    'FR': 'Fribourg',
    'SO': 'Solothurn',
    'BS': 'Basel',
    'BL': 'Liestal',
    'SH': 'Schaffhausen',
    'AR': 'Herisau',
    'AI': 'Appenzell',
    'SG': 'St. Gallen',
    'GR': 'Chur',
    'AG': 'Aarau',
    'TG': 'Frauenfeld',
    'TI': 'Bellinzona',
    'VD': 'Lausanne',
    'VS': 'Sion',
    'NE': 'Neuchâtel',
    'GE': 'Geneva',
    'JU': 'Delémont',
}

# A binary variable 'is_center' is created: 1 if the listing is located
# in the canton's capital city, 0 otherwise
df['is_center'] = df.apply(
    lambda row: 1 if row['city'] == canton_centers.get(row['canton'], '') else 0,
    axis=1
)
print(df['is_center'].value_counts())

# A boxplot compares the distribution of log price per m²
# between listings in canton centers and those outside

sns.boxplot(x='is_center', y='log_price_per_m2', data=df,
            hue='is_center', legend=False)
plt.xticks([0, 1], ['Outside Center', 'City Center'])
plt.title("Log price per m² — Center vs Outside")
plt.xlabel("")
plt.ylabel("Log price per m²")
plt.show()

"""The boxplot shows that listings in canton capital cities tend to have
# a slightly higher median log price per m² compared to listings outside
# the center. However, the distributions overlap significantly, suggesting
# that while location (center vs. outside) has some effect on rental prices,
# it is not a strong standalone predictor — consistent with the low R² value
# observed in the regression model."""


# A fourth model is added using the 'is_center' variable as a location proxy,
# and the updated R² bar chart now includes location as a third factor group
model_location = smf.ols('log_price_per_m2 ~ is_center', data=df).fit()

models = ['Economic\n(GDP)', 'Housing\nCharacteristics', 'Location\n(Center)', 'Combined']
r2_values = [model_gdp.rsquared, model_housing.rsquared,
             model_location.rsquared, model_full.rsquared]

plt.figure(figsize=(9, 5))
bars = plt.bar(models, r2_values, color=['steelblue', 'coral', 'mediumpurple', 'mediumseagreen'])
plt.ylabel("R² (Explained Variance)")
plt.title("Q3: Which factors explain rental prices per m²?")
plt.ylim(0, 0.15)

# R² values are displayed on top of each bar for readability
for bar, val in zip(bars, r2_values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.001,
             f'{val:.3f}', ha='center', fontsize=11)

plt.tight_layout()
plt.show()

"""This analysis extends the original question by introducing location as a third factor 
group.A binary variable is_center was derived from the location_text column to indicate
whether a listing is located in the canton's capital city."""