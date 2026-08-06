import joblib
import pandas as pd

model = joblib.load("../freight_cost_prediction/models/predict_freight_model.pkl")

sample_data = pd.DataFrame([
    {"Dollars": 1500},
    {"Dollars": 3200},
    {"Dollars": 4800},
    {"Dollars": 6000},
])

sample_data["predicted_freight_cost"] = model.predict(sample_data[["Dollars"]])

print(sample_data)