import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.caption("Larsen & Toubro Realty — Random Forest model trained on California Housing data")

model = joblib.load("model.joblib")
meta = json.load(open("meta.json"))
d = meta["defaults"]

col1, col2 = st.columns(2)
with col1:
    ocean = st.selectbox("Location type (ocean proximity)", meta["ocean_options"])
    income = st.slider("Median income (x $10,000)", 0.5, 15.0, round(d["median_income"], 2))
    age = st.slider("House age (years)", 1, 52, int(d["housing_median_age"]))
    lat = st.number_input("Latitude", 32.5, 42.0, round(d["latitude"], 2))
    lon = st.number_input("Longitude", -124.4, -114.3, round(d["longitude"], 2))
with col2:
    rooms = st.number_input("Total rooms (block)", 2, 40000, int(d["total_rooms"]))
    bedrooms = st.number_input("Total bedrooms (block)", 1, 7000, int(d["total_bedrooms"]))
    pop = st.number_input("Population (block)", 3, 35000, int(d["population"]))
    hh = st.number_input("Households (block)", 1, 6000, int(d["households"]))

if st.button("Predict price", type="primary"):
    row = {
        "longitude": lon, "latitude": lat, "housing_median_age": age,
        "total_rooms": rooms, "total_bedrooms": bedrooms, "population": pop,
        "households": hh, "median_income": income,
        "rooms_per_household": rooms / hh,
        "bedrooms_per_room": bedrooms / rooms,
        "people_per_household": pop / hh,
    }
    for c in meta["columns"]:
        if c.startswith("ocean_proximity_"):
            row[c] = 1 if c == f"ocean_proximity_{ocean}" else 0
    X = pd.DataFrame([row])[meta["columns"]]
    st.success(f"Estimated price: ${model.predict(X)[0]:,.0f}")

with st.expander("Model performance"):
    st.dataframe(pd.DataFrame(meta["metrics"]).round(4))
