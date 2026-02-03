# Agri Price Predictor

Agri Price Predictor is a machine learning–based web application that estimates weekly average mandi prices for selected agricultural commodities across Indian states.

The objective of this project is not to predict exact market prices, but to model historical price trends and provide a reasonable estimate for a given commodity, state, and time of year.

This project was built as part of my work in Computer Science with a specialization in Artificial Intelligence and Machine Learning, with an emphasis on real-world data handling and model limitations.

---

## What the application does

- Estimates weekly average mandi prices
- Allows selection of commodity, state, and date
- Uses the selected date only for seasonal context
- Displays prices per quintal (100 kg) with an approximate per-kg value
- Runs as an interactive web application using Streamlit

The application is intended for educational and analytical purposes, not for live trading or real-time decision making.

---

## How the prediction works

- Historical mandi price data is cleaned and standardized
- Prices are aggregated to weekly averages to reduce daily volatility
- Machine learning models are trained on historical trends
- The model estimates a typical weekly price for similar past conditions

The system learns patterns from history, not future market shocks.

---

## Technology used

- Python
- Pandas and NumPy for data processing
- Scikit-learn for model training
- Streamlit for the web interface
- Joblib for model persistence
- Git and GitHub for version control

---

## Data source

The data used in this project comes from the Government of India open data platform (data.gov.in), specifically public mandi price datasets.

Raw data files are processed locally and are not included in the repository.

---

## Project structure

agri-price-predictor/
├── app/
│   └── app.py
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── fetch_latest_data.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── requirements.txt
├── README.md
└── .gitignore

---

## Running the project locally

1. Clone the repository

git clone https://github.com/IshtiyaqNoorani/agri-price-predictor.git  
cd agri-price-predictor

2. Create and activate a virtual environment

python -m venv venv  
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Run the application

streamlit run app/app.py

The app will be available at http://localhost:8501

---

## Limitations and assumptions

- The model does not use live market data
- Predictions are based solely on historical patterns
- Sudden market shocks, weather events, or policy changes are not captured
- Output should be interpreted as approximate trends, not exact prices

These limitations reflect real constraints of historical data–driven models.

---

## Possible future improvements

- Automatic data updates using APIs
- Commodity-specific or market-level models
- Time-series approaches such as Prophet or LSTM
- Price trend visualizations
- Public cloud deployment

---

## Author

Ishtiyaq Noorani  
B.Tech – Computer Science (AI & ML)  
India

This project was developed as a learning exercise to better understand applied machine learning, data quality challenges, and the gap between theoretical models and real-world deployment.

