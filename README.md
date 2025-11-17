# Flight Fare Prediction using Machine Learning & Flask ✈️ 
📌 Project Overview

Flight ticket prices fluctuate due to multiple factors such as airline, journey date, source, destination, total stops, and more.
This project uses Machine Learning to predict flight fares and provides a simple Flask-based web interface where users can input flight details and get the predicted ticket price instantly.

This project is ideal for learning:

Machine Learning workflow

Data preprocessing and feature engineering

Model training and evaluation

Building a Flask web app

Connecting ML model with frontend (HTML + CSS)

Deploying or hosting ML projects

🚀 Features

Predicts flight ticket price based on user inputs

Clean and simple web UI

ML model built using real flight data

End-to-end integration of ML + Flask + HTML + CSS

Fully open-source and ready to deploy

🧠 Machine Learning Model

The model is trained on the Flight Fare Dataset with the following steps:

✔ Data Cleaning

Handling missing values

Dropping unnecessary columns

Converting date/time columns

✔ Feature Engineering

Extracting day, month from journeys

Separating hours/minutes from duration

Encoding categorical features

One-Hot Encoding for airlines, source & destination

✔ Model Training

Algorithms used during experimentation:

Random Forest Regression

Extra Trees Regression

Decision Tree Regression

Linear Regression

Final model used: Random Forest (saved as flight_pipeline.pkl)

🗂 Folder Structure

flight_fare_prediction/
│── app.py                    # Flask backend
│── train_model.py            # ML model building script
│── model/flight_pipeline.pkl # Trained ML model
│── requirements.txt          # Python dependencies
│── data/Data_Train.xlsx      # Dataset
│
├── templates/
│     ├── index.html          # Input page
│     └── result.html         # Output page
│
├── static/
│     └── style.css           # UI styling
│
└── README.md                 # Project documentation


🛠️ Technologies Used
🔹 Machine Learning

Python

Pandas

NumPy

Scikit-learn

Matplotlib / Seaborn (for EDA)

🔹 Web Development

Flask

HTML

CSS

📥 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/flight_fare-prediction-ml.git
cd flight_fare-prediction-ml

2️⃣ Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Flask App
python app.py

5️⃣ Open in browser
http://127.0.0.1:5000/

🖥️ Web Application Screens
🏠 Home Page

Users can input:

Airline

Source & Destination

Total Stops

Departure & Arrival Time

Journey Date

📊 Result Page

Shows the Predicted Flight Fare.

🧩 Model File

The trained machine learning model is saved using pickle:

model/flight_pipeline.pkl


This model is loaded inside app.py during prediction.

📌 Future Improvements

Deploy on Render / Railway / AWS / Heroku

Add a dropdown list for airports & airlines

Add user login and history tracking

Use real-time API flight data

Build mobile-friendly UI

👩‍💻 Author

Anjana Ajikumar
GitHub: https://github.com/Anjana-ajikumar

Project: Flight Fare Prediction using ML + Flask
