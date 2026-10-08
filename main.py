import json
import re
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
MODEL_PATH, META_PATH = BASE / "model.joblib", BASE / "meta.json"
if not MODEL_PATH.exists() or not META_PATH.exists():
    raise RuntimeError("Model files missing. Run `python train_model.py` first.")
model = joblib.load(MODEL_PATH)
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
if not {"categorical", "numeric", "area_stats", "metrics"}.issubset(meta):
    raise RuntimeError("Old model metadata found. Run `python train_model.py` first.")

app = FastAPI(title="BlockBhaav")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
AREA_INSIGHTS = meta.get("area_insights", {}).get("areas", [])


class Home(BaseModel):
    area_type: str
    availability: Literal["Ready To Move", "Immediate Possession"]
    location: str
    size: str
    total_sqft: float = Field(gt=0)
    bath: float = Field(gt=0)
    balcony: float = Field(ge=0)


class Preferences(BaseModel):
    budget_lakhs: float = Field(gt=0)
    bedrooms: int = Field(ge=1, le=10)
    minimum_sqft: float = Field(gt=0)
    maximum_metro_km: float = Field(gt=0, le=30)
    availability: Literal["Ready To Move", "Immediate Possession"]
    area_type: str
    priority: Literal["Balanced", "Lower price", "More space", "Near metro"]


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/meta")
def get_meta():
    return meta


@app.post("/api/predict")
def predict(home: Home):
    cols = meta["categorical"] + meta["numeric"]
    row = pd.DataFrame([home.model_dump()], columns=cols)
    lakhs = max(0.0, float(model.predict(row)[0]))
    return {"price_lakhs": lakhs, "price_rupees": lakhs * 100_000}


def context_for(area):
    key = re.sub(r"\s+", " ", area).strip().casefold()
    info = next((a for a in AREA_INSIGHTS if a["area"].casefold() == key), None)
    history = meta.get("history", {}).get(key, {"records": [], "cagr_pct": None, "forecast_3y_lakhs": None})
    context = meta.get("area_context", {}).get(key, {})
    metro = meta.get("location_coords", {}).get(area, meta.get("area_coords", {}).get(area, {}))
    return info, history, context, metro


@app.get("/api/area")
def get_area(area: str):
    key = re.sub(r"\s+", " ", area).strip().casefold()
    local = next((x for x in meta["area_stats"] if x["area"].casefold() == key), None)
    info = next((x for x in AREA_INSIGHTS if x["area"].casefold() == key), None)
    return {
        "area": area,
        "housing": meta.get("area_insight_stats", {}).get(key, local),
        "reference": info,
        "history": meta.get("history", {}).get(key, {"records": [], "cagr_pct": None, "forecast_3y_lakhs": None}),
        "context": meta.get("area_context", {}).get(key, {}),
        "metro": meta.get("location_coords", {}).get(area, meta.get("area_coords", {}).get(area, {})),
        "reference_note": meta.get("area_insights", {}).get("source_note", ""),
    }


@app.post("/api/recommend")
def recommend(p: Preferences):
    options = []
    for s in meta["area_stats"]:
        if s["median_price_lakhs"] > p.budget_lakhs or s["median_sqft"] < p.minimum_sqft:
            continue
        if abs(s["median_bhk"] - p.bedrooms) > 1.1:
            continue
        if s["availability"].casefold() != p.availability.casefold():
            continue
        if p.area_type != "Any" and s["area_type"].casefold() != p.area_type.casefold():
            continue
        metro = meta.get("location_coords", {}).get(s["area"], meta.get("area_coords", {}).get(s["area"], {}))
        distance = metro.get("distance_km")
        if distance is None or distance > p.maximum_metro_km:
            continue
        price_fit = max(0, 1 - s["median_price_lakhs"] / p.budget_lakhs)
        space_fit = min(s["median_sqft"] / max(p.minimum_sqft * 1.6, 1), 1)
        metro_fit = max(0, 1 - distance / p.maximum_metro_km)
        scores = {
            "Balanced": .45 * price_fit + .25 * space_fit + .30 * metro_fit,
            "Lower price": .70 * price_fit + .15 * space_fit + .15 * metro_fit,
            "More space": .20 * price_fit + .65 * space_fit + .15 * metro_fit,
            "Near metro": .20 * price_fit + .15 * space_fit + .65 * metro_fit,
        }
        ref, history, context, _ = context_for(s["area"])
        options.append({
            **s, "metro": metro, "score": scores[p.priority],
            "reference": ref, "history": history, "context": context,
        })
    options.sort(key=lambda x: (x["score"], x["listing_count"]), reverse=True)
    return {"recommendations": options[:5], "message": "" if options else "No localities match all choices. Try a higher budget, more metro distance, or a smaller home."}
