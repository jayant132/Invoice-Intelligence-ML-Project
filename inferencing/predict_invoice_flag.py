import joblib
import pandas as pd

model = joblib.load("../invoice_flagging/models/random_forest.pkl")
scaler = joblib.load("../invoice_flagging/models/scaler.pkl")

sample_data = pd.DataFrame([
    {"Invoice_Quantity": 120, "Invoice_Dollars": 9500, "Freight_Cost": 150, "Total_Brands": 3, "Total_Item_Quantity": 118, "Total_Item_Dollars": 4480, "Avg_Receiving_Delay": 3},
    {"Invoice_Quantity": 80, "Invoice_Dollars": 6200, "Freight_Cost": 300, "Total_Brands": 2, "Total_Item_Quantity": 80, "Total_Item_Dollars": 5500, "Avg_Receiving_Delay": 4},
    {"Invoice_Quantity": 50, "Invoice_Dollars": 2000, "Freight_Cost": 75, "Total_Brands": 1, "Total_Item_Quantity": 50, "Total_Item_Dollars": 1990, "Avg_Receiving_Delay": 2},
])

sample_data["Dollar_Gap"] = (sample_data["Invoice_Dollars"] - sample_data["Total_Item_Dollars"]).abs()

X_scaled = scaler.transform(sample_data)
sample_data["predicted_flag"] = model.predict(X_scaled)

print(sample_data)