import streamlit as st
import pickle
import pandas as pd
from catboost import CatBoostRegressor


# =============================================================
# LOAD MODEL AND ARTIFACTS
# =============================================================

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model('models/catboost_rent_model.cbm')
    return model

@st.cache_resource
def load_artifacts():
    with open('models/city_canton_map.pkl', 'rb') as f:
        city_canton_map = pickle.load(f)
    with open('models/top_cities.pkl', 'rb') as f:
        top_cities = pickle.load(f)
    return city_canton_map, top_cities

model           = load_model()
city_canton_map, top_cities = load_artifacts()

# =============================================================
# UI
# =============================================================

st.title("🏠 Swiss Rental Price Predictor")
st.markdown("Estimate monthly rent for apartments in Switzerland.")

st.divider()

# City seçimi
city_options = ['Other'] + sorted(top_cities)
city = st.selectbox("City", options=city_options)

# Canton — otomatik veya manuel
if city != 'Other':
    canton = city_canton_map[city]
    st.info(f"Canton: **{canton}**")
else:
    CANTONS = [
        "ZH","BE","LU","UR","SZ","OW","NW","GL","ZG","FR",
        "SO","BS","BL","SH","AR","AI","SG","GR","AG","TG",
        "TI","VD","VS","NE","GE","JU"
    ]
    canton = st.selectbox("Canton", options=sorted(CANTONS))

# Diğer inputlar
living_area = st.number_input("Living area (m²)", min_value=10, max_value=500, value=80)
rooms       = st.selectbox("Rooms", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])

st.divider()

# =============================================================
# PREDICTION
# =============================================================

if st.button("Predict Rent"):
    input_df = pd.DataFrame([{
        'canton'        : canton,
        'city_grouped'  : city,
        'living_area_m2': living_area,
        'rooms'         : rooms
    }])

    prediction = model.predict(input_df)[0]
    prediction = max(0, round(prediction, 0))

    st.success(f"### Estimated monthly rent: CHF {prediction:,.0f}")
    st.caption("⚠️ Prediction is based on listings from immobilier.ch (March 2026). "
               "Actual rent may vary depending on building age, floor, and condition.")



