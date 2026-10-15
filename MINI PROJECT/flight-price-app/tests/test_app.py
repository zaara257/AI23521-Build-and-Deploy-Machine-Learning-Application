"""Tests for the feature code and the API.   Run:  pytest -q

Needs models/flight_model.joblib, so run `python train.py` first.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app  # noqa: E402
from features import FEATURES, clean_raw, duration_to_minutes, engineer_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

GOOD = {
    "airline": "IndiGo", "source": "Bengaluru", "destination": "Delhi",
    "journey_date": "2019-04-19", "dep_time": "07:30", "duration_mins": 170,
    "stops": 0, "extras": "No info",
}


@pytest.fixture()
def client():
    return app.test_client()


# ---------------------------------------------------------------- feature engineering
def test_duration_parsing():
    assert duration_to_minutes("2h 50m") == 170
    assert duration_to_minutes("19h") == 1140
    assert duration_to_minutes("5m") == 5


def test_festival_and_peak_features():
    row = pd.DataFrame([{
        "Airline": "IndiGo", "Source": "Delhi", "Destination": "Cochin", "Additional_Info": "No info",
        "Journey_Date": "2019-03-21", "Dep_Time": "08:15", "Duration_Mins": 300, "Total_Stops": 1,
    }])
    f = engineer_features(row).iloc[0]
    assert list(engineer_features(row).columns) == FEATURES
    assert f["Days_To_Festival"] == 0 and f["Near_Festival"] == 1   # Holi
    assert f["Is_Peak_Hour"] == 1                                    # 08:15 departure
    assert f["Is_Weekend"] == 0                                      # Thursday
    assert f["Is_Summer_Peak"] == 0                                  # before 1 May


def test_cleaning_removes_bad_rows():
    raw = pd.read_excel(ROOT / "data" / "Data_Train.xlsx")
    tidy, log = clean_raw(raw)
    assert tidy.isna().sum().sum() == 0
    assert log["duplicates_dropped"] > 0 and log["bad_duration_dropped"] == 1
    assert (tidy["Duration_Mins"] >= 30).all()
    assert set(tidy["Source"]) == {"Bengaluru", "Chennai", "Delhi", "Kolkata", "Mumbai"}


# ---------------------------------------------------------------- API: usability
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_form_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Predict price" in r.data


def test_meta_has_dropdown_options(client):
    meta = client.get("/meta").get_json()
    assert "IndiGo" in meta["airlines"] and "Delhi" in meta["routes"]["Bengaluru"]


def test_predict_ok(client):
    r = client.post("/predict", json=GOOD)
    body = r.get_json()
    assert r.status_code == 200
    assert 1000 < body["predicted_price_inr"] < 60000
    assert body["factors"]["nearest_festival"] == "Good Friday"
    assert body["factors"]["is_peak_hour"] is True


@pytest.mark.parametrize("change, expected", [
    ({"airline": "Emirates"}, "Unknown airline"),
    ({"destination": "Cochin"}, "No data for Bengaluru -> Cochin"),
    ({"journey_date": "2026-12-25"}, "journey_date must be between"),
    ({"journey_date": "25/12/2019"}, "YYYY-MM-DD"),
    ({"dep_time": "25:99"}, "HH:MM"),
    ({"duration_mins": 5}, "duration_mins must be between"),
    ({"stops": 9}, "stops must be between"),
    ({"extras": "Free champagne"}, "Unknown extras"),
])
def test_predict_rejects_bad_input(client, change, expected):
    r = client.post("/predict", json={**GOOD, **change})
    assert r.status_code == 400
    assert any(expected in e for e in r.get_json()["errors"])


def test_predict_reports_all_missing_fields(client):
    r = client.post("/predict", json={})
    assert r.status_code == 400 and len(r.get_json()["errors"]) >= 6


def test_predict_without_json_body(client):
    assert client.post("/predict", data="not json").status_code == 400


# ---------------------------------------------------------------- API: sanity of the predictions
def test_prediction_responds_to_inputs(client):
    base = client.post("/predict", json=GOOD).get_json()["predicted_price_inr"]
    other = client.post("/predict", json={**GOOD, "airline": "Jet Airways", "stops": 2,
                                          "duration_mins": 1200}).get_json()["predicted_price_inr"]
    assert other != base


# ---------------------------------------------------------------- accuracy on held-out data
def test_model_accuracy_meets_threshold():
    """The saved model's recorded held-out MAE/R2 must stay within an acceptable range."""
    meta = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["metrics"]["mae"] < 1000
    assert meta["metrics"]["r2"] > 0.85
