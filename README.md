# Smart Flight Fare Predictor

## Setup
1. Create virtual env:
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt
   #pip install flask
   #pip install pandas
   #pip install scikit-learn


3. Put dataset Data_Train.csv into data/ (download from Kaggle).

4. Train model:
   python train_model.py
   -> this creates model/flight_pipeline.pkl

5. Run Flask app:
   python app.py
   Open http://127.0.0.1:5000 in browser.

## Notes
- Ensure train_model.py runs without errors before running app.
- For deployment use Render / Railway / Heroku and create environment variables if needed.
