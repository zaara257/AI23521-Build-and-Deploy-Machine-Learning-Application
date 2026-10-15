"""Flask REST API + web form for the flight price model.

    python app.py                      # development server on http://localhost:5000
    gunicorn -b 0.0.0.0:5000 app:app   # what the Docker image runs

Endpoints
    GET  /          web form
    GET  /health    liveness check
    GET  /meta      dropdown options, valid date range, model metrics
    POST /predict   JSON in -> predicted price out
"""
import json
from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from festivals import FESTIVAL_WINDOW_DAYS, SUMMER_PEAK_START, nearest_festival
from features import engineer_features

ROOT = Path(__file__).parent
MODEL = joblib.load(ROOT / "models" / "flight_model.joblib")
META = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))

DATE_MIN = date.fromisoformat(META["date_range"][0])
DATE_MAX = date.fromisoformat(META["date_range"][1])
DUR_MIN, DUR_MAX = 60, 2880  # minutes (training data runs from 75 min to ~47 h)

app = Flask(__name__)


def validate(payload):
    """Check the request body. Returns (clean_inputs, errors)."""
    errors = []
    if not isinstance(payload, dict):
        return None, ["Send a JSON object."]

    def get(name):
        value = payload.get(name)
        if value in (None, ""):
            errors.append(f"'{name}' is required.")
        return value

    airline, source, dest = get("airline"), get("source"), get("destination")
    when, dep, dur, stops = get("journey_date"), get("dep_time"), get("duration_mins"), get("stops")
    extras = payload.get("extras") or "No info"
    if errors:
        return None, errors

    if airline not in META["airlines"]:
        errors.append(f"Unknown airline '{airline}'. Choose one of: {', '.join(META['airlines'])}.")
    if source not in META["routes"]:
        errors.append(f"Unknown source '{source}'. Choose one of: {', '.join(META['routes'])}.")
    elif dest not in META["routes"][source]:
        errors.append(f"No data for {source} -> {dest}. Valid destinations from {source}: "
                      f"{', '.join(META['routes'][source])}.")
    if extras not in META["extras"]:
        errors.append(f"Unknown extras '{extras}'. Choose one of: {', '.join(META['extras'])}.")

    try:
        journey = datetime.strptime(str(when), "%Y-%m-%d").date()
        if not DATE_MIN <= journey <= DATE_MAX:
            errors.append(f"journey_date must be between {DATE_MIN} and {DATE_MAX} "
                          "(the period the model was trained on).")
    except ValueError:
        errors.append("journey_date must look like YYYY-MM-DD.")
        journey = None

    try:
        hh, mm = (int(p) for p in str(dep).split(":"))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError
        dep = f"{hh:02d}:{mm:02d}"
    except ValueError:
        errors.append("dep_time must look like HH:MM (24-hour).")

    try:
        dur = int(dur)
        if not DUR_MIN <= dur <= DUR_MAX:
            errors.append(f"duration_mins must be between {DUR_MIN} and {DUR_MAX}.")
    except (TypeError, ValueError):
        errors.append("duration_mins must be a whole number of minutes.")

    try:
        stops = int(stops)
        if not 0 <= stops <= 3:
            errors.append("stops must be between 0 and 3.")
    except (TypeError, ValueError):
        errors.append("stops must be a whole number between 0 and 3.")

    if errors:
        return None, errors
    return {"Airline": airline, "Source": source, "Destination": dest, "Additional_Info": extras,
            "Journey_Date": journey, "Dep_Time": dep, "Duration_Mins": dur, "Total_Stops": stops}, []


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok", model=META["best_model"])


@app.get("/meta")
def meta():
    return jsonify(
        model=META["best_model"],
        metrics=META["metrics"],
        airlines=META["airlines"],
        extras=META["extras"],
        routes=META["routes"],
        date_range=META["date_range"],
        festivals=META["festivals"],
    )


@app.post("/predict")
def predict():
    inputs, errors = validate(request.get_json(silent=True))
    if errors:
        return jsonify(errors=errors), 400

    features = engineer_features(pd.DataFrame([inputs]))
    price = float(MODEL.predict(features)[0])

    days_away, fest_name = nearest_festival(inputs["Journey_Date"])
    row = features.iloc[0]
    return jsonify(
        predicted_price_inr=int(round(price)),
        typical_error_inr=int(round(META["metrics"]["mae"])),
        model=META["best_model"],
        factors={
            "is_weekend": bool(row["Is_Weekend"]),
            "is_peak_hour": bool(row["Is_Peak_Hour"]),
            "near_festival": bool(row["Near_Festival"]),
            "nearest_festival": fest_name,
            "days_to_festival": days_away,
            "festival_window_days": FESTIVAL_WINDOW_DAYS,
            "is_summer_peak": bool(inputs["Journey_Date"] >= SUMMER_PEAK_START),
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
