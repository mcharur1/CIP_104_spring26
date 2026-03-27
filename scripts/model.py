"""
model.py
========
Rental Price Prediction Model for Swiss Cantons
------------------------------------------------
This script builds and evaluates two gradient boosting models
(LightGBM and CatBoost) to predict monthly rental prices (CHF)
for apartments and houses across Swiss cantons.

WHY LIGHTGBM AND CATBOOST?
---------------------------
Both models are gradient boosting frameworks that outperform
traditional linear regression when:
  - Relationships between features and target are non-linear
  - Categorical variables (canton, city) are present
  - Dataset size is moderate (thousands of rows)

LightGBM  : Fast training, efficient with large datasets,
            requires manual encoding of categorical variables.
CatBoost  : Handles categorical variables natively without
            encoding, often achieves better accuracy with
            less hyperparameter tuning. Preferred when
            categorical features dominate, as in our case
            (canton + city_grouped).

WHY FILTER TO NORMAL SEGMENT (price_chf <= 5000)?
--------------------------------------------------
Exploratory analysis revealed extreme outliers in the price
distribution (max: 230,000 CHF). These luxury properties
represent a fundamentally different market segment and
cannot be modelled alongside standard rentals. Including
them severely degrades model performance (R² dropped from
~0.75 to ~0.10). Since 96% of listings fall within the
normal segment, filtering is justified and causes minimal
data loss.
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# =============================================================
# 1. LOAD AND PRE-PROCESS DATA
# =============================================================

# Load GDP data
df_gdp = pd.read_csv('../data/processed/canton_gdp_2022_clean.csv')
df_gdp = df_gdp.rename(columns={'Canton': 'canton_name', '2022': 'canton_gdp'})

CANTONS = {
    "ZH": "Zurich", "BE": "Berne", "LU": "Lucerne", "UR": "Uri",
    "SZ": "Schwyz", "OW": "Obwalden", "NW": "Nidwalden", "GL": "Glarus",
    "ZG": "Zug", "FR": "Fribourg", "SO": "Solothurn", "BS": "Basel-Stadt",
    "BL": "Basel-Landschaft", "SH": "Schaffhausen", "AR": "Appenzell A. Rh.",
    "AI": "Appenzell I. Rh.", "SG": "St. Gallen", "GR": "Graubünden",
    "AG": "Aargau", "TG": "Thurgau", "TI": "Ticino", "VD": "Vaud",
    "VS": "Valais", "NE": "Neuchâtel", "GE": "Geneva", "JU": "Jura",
    "CH": "Switzerland"
}

df_cantons = pd.DataFrame(list(CANTONS.items()), columns=['canton', 'canton_name'])
df_gdp = pd.merge(df_gdp, df_cantons, on='canton_name', how='left')

# Load listing data
df_listings = pd.read_csv('../data/processed/immobilier_all_cantons_allpages_snapshot_p20.csv')

# Merge listing and GDP data
df = pd.merge(df_listings, df_gdp, on='canton', how='left')
df['date_scraped'] = pd.to_datetime(df['date_scraped'])


# =============================================================
# 2. HANDLE MISSING VALUES
# =============================================================

# Impute living_area_m2 and rooms using canton-level median
# to account for regional differences
for col in ['living_area_m2', 'rooms']:
    df[col] = df.groupby('canton')[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())


# =============================================================
# 3. FEATURE ENGINEERING
# =============================================================

# Derived price features (used in EDA/regression, excluded from ML to avoid leakage)
df['price_per_m2']     = df['price_chf'] / df['living_area_m2']
df['log_price']        = np.log(df['price_chf'])
df['log_price_per_m2'] = np.log(df['price_per_m2'])

# Extract city from location_text (remove 4-digit postal code prefix)
df['city'] = df['location_text'].str.replace(r'^\d{4}\s*', '', regex=True).str.strip()

# Binary indicator: is the listing in the canton's capital city?
canton_centers = {
    'ZH': 'Zürich', 'BE': 'Bern', 'LU': 'Luzern', 'UR': 'Altdorf',
    'SZ': 'Schwyz', 'OW': 'Sarnen', 'NW': 'Stans', 'GL': 'Glarus',
    'ZG': 'Zug', 'FR': 'Fribourg', 'SO': 'Solothurn', 'BS': 'Basel',
    'BL': 'Liestal', 'SH': 'Schaffhausen', 'AR': 'Herisau',
    'AI': 'Appenzell', 'SG': 'St. Gallen', 'GR': 'Chur', 'AG': 'Aarau',
    'TG': 'Frauenfeld', 'TI': 'Bellinzona', 'VD': 'Lausanne',
    'VS': 'Sion', 'NE': 'Neuchâtel', 'GE': 'Geneva', 'JU': 'Delémont',
}
df['is_center'] = df.apply(
    lambda row: 1 if row['city'] == canton_centers.get(row['canton'], '') else 0,
    axis=1
)

# =============================================================
# 4. PRICE DISTRIBUTION — NORMAL SEGMENT FILTER
# =============================================================
# Rental prices contain extreme outliers (max: 230,000 CHF)
# representing luxury properties that form a separate market.
# These outliers collapse model performance when included.
# Since 96% of listings fall at or below 5,000 CHF/month,
# filtering to this segment causes minimal data loss while
# substantially improving prediction accuracy.

print(df['price_chf'].describe())
print("\nVery expensive listings (5,000+ CHF) :", (df['price_chf'] > 5000).sum())
print("Very expensive listings (10,000+ CHF):", (df['price_chf'] > 10000).sum())
print("Maximum price                         :", df['price_chf'].max())

df_normal = df[df['price_chf'] <= 5000].copy()

print(f"\nNormal segment (<=5,000 CHF) : {len(df_normal)} listings "
      f"({len(df_normal)/len(df)*100:.1f}% of total)")
print(f"Excluded luxury listings     : {len(df) - len(df_normal)} listings")
print(f"Total listings               : {len(df)}")


# =============================================================
# 5. CITY GROUPING
# =============================================================
# 994 unique cities exist in the dataset, but 906 have fewer
# than 10 observations. Rare cities provide no reliable signal
# for the model. Cities with 20+ listings are kept as-is;
# all others are grouped into "Other".

city_counts      = df_normal['city'].value_counts()
top_cities       = city_counts[city_counts >= 20].index.tolist()
df_normal['city_grouped'] = df_normal['city'].apply(
    lambda x: x if x in top_cities else 'Other'
)

print(f"\nTotal unique cities          : {df_normal['city'].nunique()}")
print(f"Cities with 20+ listings     : {len(top_cities)}")
print(f"city_grouped unique values   : {df_normal['city_grouped'].nunique()}")


# =============================================================
# 6. CITY → CANTON MAPPING
# =============================================================
# Maps each known city to its most frequent canton.
# Used in the Streamlit app to auto-fill canton when city is selected.

city_canton_map = (
    df_normal[df_normal['city_grouped'] != 'Other']
    .groupby('city_grouped')['canton']
    .agg(lambda x: x.mode()[0])
    .to_dict()
)


# =============================================================
# 7. TRAIN / TEST SPLIT
# =============================================================

FEATURES = ['canton', 'city_grouped', 'living_area_m2', 'rooms']
TARGET   = 'price_chf'

df_model = df_normal[FEATURES + [TARGET]].dropna()
X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain set : {len(X_train)} listings")
print(f"Test set  : {len(X_test)} listings")


# =============================================================
# 8. EVALUATION FUNCTION
# =============================================================

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  MAE  : {mae:,.0f} CHF   (average absolute error)")
    print(f"  RMSE : {rmse:,.0f} CHF")
    print(f"  R²   : {r2:.4f}        (variance explained)")
    return {'Model': name, 'MAE (CHF)': round(mae),
            'RMSE (CHF)': round(rmse), 'R²': round(r2, 4)}


# =============================================================
# 9. LIGHTGBM
# =============================================================

X_train_lgb = X_train.copy()
X_test_lgb  = X_test.copy()
for col in ['canton', 'city_grouped']:
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_test_lgb[col]  = X_test_lgb[col].astype('category')

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train_lgb, y_train)
lgb_preds   = lgb_model.predict(X_test_lgb)
lgb_results = evaluate("LightGBM", y_test, lgb_preds)


# =============================================================
# 10. CATBOOST
# =============================================================

cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0,
    cat_features=['canton', 'city_grouped']
)
cat_model.fit(X_train, y_train)
cat_preds   = cat_model.predict(X_test)
cat_results = evaluate("CatBoost", y_test, cat_preds)


# =============================================================
# 11. MODEL COMPARISON
# =============================================================

print("\n")
results_df = pd.DataFrame([lgb_results, cat_results])
print(results_df.to_string(index=False))


# =============================================================
# 12. ACTUAL VS PREDICTED PLOT
# =============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, name in zip(axes,
                            [lgb_preds, cat_preds],
                            ['LightGBM', 'CatBoost']):
    ax.scatter(y_test, preds, alpha=0.3, s=15, color='steelblue')
    lim = max(y_test.max(), preds.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1, label='Perfect prediction')
    ax.set_xlabel("Actual price (CHF)")
    ax.set_ylabel("Predicted price (CHF)")
    ax.set_title(name)
    ax.legend(fontsize=9)
plt.suptitle("Actual vs Predicted — price_chf (normal segment)")
plt.tight_layout()
plt.show()


# =============================================================
# 13. FEATURE IMPORTANCE
# =============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

lgb.plot_importance(lgb_model, ax=axes[0], max_num_features=10,
                    importance_type='gain',
                    title='LightGBM — Feature Importance (gain)')

cat_importance = pd.Series(
    cat_model.get_feature_importance(),
    index=FEATURES
).sort_values()
cat_importance.plot(kind='barh', ax=axes[1], color='coral')
axes[1].set_title('CatBoost — Feature Importance')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.show()

# =============================================================
# 14. SAVE BEST MODEL (CATBOOST)
# =============================================================
# CatBoost selected as the final model based on superior MAE,
# RMSE, and R² on the test set.

os.makedirs('../models', exist_ok=True)

cat_model.save_model('../models/catboost_rent_model.cbm')
print("Model saved     → ../models/catboost_rent_model.cbm")

# city_canton_map.pkl
#   A dict mapping each known city to its most frequent canton.
#   Used in Streamlit: when user selects a city, canton auto-fills.
#   If user selects "Other", canton dropdown appears manually.
#   Example: {'Zürich': 'ZH', 'Basel': 'BS', 'Geneva': 'GE', ...}
#   Load:  city_canton_map = pickle.load(open('../models/city_canton_map.pkl', 'rb'))

with open('../models/city_canton_map.pkl', 'wb') as f:
    pickle.dump(city_canton_map, f)
print("Mapping saved   → ../models/city_canton_map.pkl")

# top_cities.pkl
#   A list of 30 cities with 20+ listings in the normal segment.
#   Used to populate the city dropdown in Streamlit.
#   Everything outside this list is labelled "Other".
#   Example: ['Zürich', 'Basel', 'Geneva', 'Lausanne', ...]
#   Load:  top_cities = pickle.load(open('../models/top_cities.pkl', 'rb'))

with open('../models/top_cities.pkl', 'wb') as f:
    pickle.dump(top_cities, f)
print("Top cities saved → ../models/top_cities.pkl")


# =============================================================
# 15. SAMPLE PREDICTION
# =============================================================
# A random sample of 5 listings from the test set is used to
# visually verify that the model produces reasonable predictions.
# Actual price, predicted price, and error (CHF) are shown side by side.


sample = X_test.copy()
sample['actual_price']    = y_test.values
sample['predicted_price'] = cat_model.predict(X_test).round(0)
sample['error_chf']       = (sample['predicted_price'] - sample['actual_price']).round(0)

print("\nSample predictions (5 random listings from test set):")
print(sample.sample(5, random_state=42).to_string())


"""The model performs well for cantons with sufficient observations (MAE ~100-200 CHF).
However, for cantons with limited data such as AR (32 listings), AI (4 listings), or OW
(5 listings), prediction accuracy drops significantly. This is an inherent limitation of the
dataset rather than the model itself — more listings from underrepresented cantons would 
substantially improve performance."""

# Sample predictions (5 random listings from test set):
#     canton city_grouped  living_area_m2  rooms  actual_price  predicted_price  error_chf
# 1513     SO        Other            67.0    2.5        1400.0           1242.0     -158.0
# 1999     AR        Other           130.0    5.5        3150.0           1916.0    -1234.0
# 154      ZH        Other            38.0    1.5        1160.0           1661.0      501.0
# 2578     TG        Other            91.0    4.5        1490.0           1460.0      -30.0
# 2026     SG        Other            72.0    2.5        1290.0           1216.0      -74.0