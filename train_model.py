# train_model.py
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ----- CONFIG -----
DATA_PATH = "data\Data_Train.xlsx"   
SAVE_PATH = "model/flight_pipeline.pkl"
os.makedirs("model", exist_ok=True)

# ----- LOAD -----
df = pd.read_excel(DATA_PATH)

# ----- BASIC PREPROCESSING -----
# Drop rows with missing target or extremely malformed rows
df = df.dropna(subset=['Price'])

# Extract date/time features
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True, errors='coerce')
df['Journey_day'] = df['Date_of_Journey'].dt.day
df['Journey_month'] = df['Date_of_Journey'].dt.month

df['Dep_Time'] = pd.to_datetime(df['Dep_Time'], format='%H:%M', errors='coerce')
df['Dep_hour'] = df['Dep_Time'].dt.hour
df['Dep_min'] = df['Dep_Time'].dt.minute

df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'], format='%H:%M', errors='coerce')
df['Arrival_hour'] = df['Arrival_Time'].dt.hour
df['Arrival_min'] = df['Arrival_Time'].dt.minute

# Duration: convert to total minutes if present like '2h 50m'
def convert_duration(x):
    if pd.isnull(x): return np.nan
    hrs = 0
    mins = 0
    parts = x.split()
    for p in parts:
        if 'h' in p:
            try: hrs = int(p.replace('h',''))
            except: hrs = 0
        if 'm' in p:
            try: mins = int(p.replace('m',''))
            except: mins = 0
    return hrs*60 + mins

df['Duration_mins'] = df['Duration'].apply(convert_duration)

# Clean Total_Stops to numeric
def clean_stops(x):
    if x == 'non-stop' or x=='non stop' or x==0 or x=='0':
        return 0
    if isinstance(x, str):
        if 'non' in x.lower(): return 0
        if '2' in x: return 2
        if '3' in x: return 3
        if '4' in x: return 4
        # handle '1 stop'
        try:
            return int(x.split()[0])
        except:
            return np.nan
    return x

df['Total_Stops'] = df['Total_Stops'].replace('non-stop', '0')
df['Total_Stops'] = df['Total_Stops'].apply(clean_stops)

# Drop rows with any NaNs produced above
df = df.dropna(subset=['Journey_day','Journey_month','Dep_hour','Dep_min','Arrival_hour','Arrival_min','Duration_mins','Total_Stops'])

# Select features and target
FEATURES = ['Airline','Source','Destination','Total_Stops','Journey_day','Journey_month','Dep_hour','Dep_min','Arrival_hour','Arrival_min','Duration_mins']
TARGET = 'Price'

X = df[FEATURES]
y = df[TARGET].astype(float)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Preprocessing: one-hot for categorical, scaler for numeric
cat_features = ['Airline','Source','Destination']
num_features = ['Total_Stops','Journey_day','Journey_month','Dep_hour','Dep_min','Arrival_hour','Arrival_min','Duration_mins']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
        ('num', StandardScaler(), num_features)
    ]
)

pipeline = Pipeline([
    ('preproc', preprocessor),
    ('rfr', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))
])

# Fit model
print("Training model...")
pipeline.fit(X_train, y_train)

# Eval
preds = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print(f"Test RMSE: {rmse:.2f}, R2: {r2:.4f}")

# Save pipeline
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(pipeline, f)
print(f"Saved pipeline to {SAVE_PATH}")
