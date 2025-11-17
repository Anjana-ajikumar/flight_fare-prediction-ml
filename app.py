# app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
MODEL_PATH = "model/flight_pipeline.pkl"

# Load pipeline
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Run train_model.py first to create model/flight_pipeline.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Pre-fill dropdown choices by reading columns used during training
# For simplicity, we hardcode typical values (students can auto-generate these from training data)
AIRLINES = ["IndiGo","Air India","Jet Airways","SpiceJet","Multiple carriers","Vistara","GoAir","Trujet","Air Asia"]
SOURCES = ["Delhi","Kolkata","Mumbai","Chennai","Bangalore"]
DESTINATIONS = ["Cochin","Hyderabad","New Delhi","Kolkata","Bangalore","Chennai","Delhi","Mumbai"]

@app.route('/')
def home():
    return render_template('index.html', airlines=AIRLINES, sources=SOURCES, destinations=DESTINATIONS)

@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    airline = form.get('airline')
    source = form.get('source')
    destination = form.get('destination')
    stops = form.get('stops')
    date = form.get('date')          # YYYY-MM-DD
    dep_time = form.get('dep_time')  # HH:MM
    arr_time = form.get('arr_time')  # HH:MM (optional)
    duration = form.get('duration')  # optional (like '2h 50m') - if empty we will compute from times

    # Parse numeric fields
    try:
        total_stops = int(stops)
    except:
        total_stops = 0

    # Date parsing
    jd = pd.to_datetime(date, format="%Y-%m-%d", errors='coerce')
    journey_day = int(jd.day)
    journey_month = int(jd.month)

    # Dep time
    dep = pd.to_datetime(dep_time, format="%H:%M", errors='coerce')
    dep_hour = int(dep.hour)
    dep_min = int(dep.minute)

    # Arrival
    arrival = pd.to_datetime(arr_time, format="%H:%M", errors='coerce')
    arrival_hour = int(arrival.hour)
    arrival_min = int(arrival.minute)

    # Duration in mins
    def calc_duration(dep, arr):
        diff = (arr - dep).total_seconds() / 60.0
        if diff < 0:
            diff += 24*60
        return diff
    duration_mins = calc_duration(dep, arrival)

    # Build dataframe for single sample
    sample = pd.DataFrame([{
        'Airline': airline,
        'Source': source,
        'Destination': destination,
        'Total_Stops': total_stops,
        'Journey_day': journey_day,
        'Journey_month': journey_month,
        'Dep_hour': dep_hour,
        'Dep_min': dep_min,
        'Arrival_hour': arrival_hour,
        'Arrival_min': arrival_min,
        'Duration_mins': duration_mins
    }])

    # Predict
    pred = model.predict(sample)[0]
    pred = round(float(pred), 2)

    return render_template('result.html', prediction_text=f"Predicted Flight Fare: ₹{pred}")

if __name__ == "__main__":
    app.run(debug=True)
