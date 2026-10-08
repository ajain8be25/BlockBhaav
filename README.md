# BlockBhaav

BlockBhaav is a Bengaluru home-price exploration demo with Prediction, Statistics, and Recommendations pages.

## Start the app

Open a terminal in the folder containing `main.py` and run:

```text
python -m pip install -r requirements.txt
python train_model.py
python -m uvicorn main:app --reload
```

Open <http://127.0.0.1:8000>. Keep the terminal running. Run `python train_model.py` again after changing the property CSV or optional context files.

## What's included

- The interactive Bengaluru map has approximate center pins for the 20 areas on the Statistics page. The main Prediction dropdown contains every locality in the property CSV.
- Statistics only lists the 20 areas in `area_insights.json`, as requested. It shows listing summaries, the supplied area's AQI/crime/metro notes, model-independent market snapshots, and historical trend analysis when dated data is provided.
- Recommendations asks seven questions and ranks localities by the listing data and approximate metro proximity.
- The compact model report compares Ridge Regression, Random Forest, and Extra Trees using RMSE, MAE, MAPE, R², and 5-fold CV R². The model with the strongest CV R² is saved for estimates.

## Data limits and adding data

The supplied property CSV has listing prices but no sale dates, latitude/longitude, crime counts by locality, or AQI readings. Therefore it cannot support a real price CAGR/forecast or an official area crime rate. `area_insights.json` contains the rough context in the user-provided table; it is explicitly marked as unverified and not live. Approximate metro distances use approximate area/station centers and are straight-line distances, not road or walking routes.

To add property rows/localities, put same-schema Bengaluru listing CSVs in `additional_data/` and retrain. Required headers are `area_type`, `availability`, `location`, `size`, `total_sqft`, `bath`, `balcony`, and `price` (₹ lakh). Only Ready To Move and Immediate Possession records are used.

To enable year-over-year growth and a simple 3-year trend projection, add real annual market observations to `historical_prices.csv` with columns `area,year,avg_price_lakhs` and at least two different years per area. The projection is a trend scenario, not a guarantee.

To replace the rough AQI/crime notes with measured values, add rows to `area_context.csv`: `area,aqi,aqi_date,aqi_station,crime_context,crime_rate_per_1000,crime_year`. Add a source and date for each observation. Do not treat city-wide case totals as locality crime rates.

The map uses Leaflet and OpenStreetMap tiles, so it needs internet access and retains OpenStreetMap attribution. Without map access, use the locality dropdown.
