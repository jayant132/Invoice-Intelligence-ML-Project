import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Invoice Intelligence", page_icon="🧾", layout="centered")


@st.cache_resource
def load_models():
    flag_model = joblib.load("models/random_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    freight_model = joblib.load("models/predict_freight_model.pkl")
    return flag_model, scaler, freight_model


flag_model, scaler, freight_model = load_models()

st.title("🧾 Invoice Intelligence")
st.caption("ML-powered freight cost prediction & invoice fraud detection")

tab1, tab2 = st.tabs(["🚩 Invoice Flagging", "📦 Freight Cost Prediction"])

with tab1:
    st.subheader("Check if an invoice looks risky")
    st.write("Enter invoice and matching purchase-order details to see if this invoice would be flagged.")

    col1, col2 = st.columns(2)

    with col1:
        invoice_quantity = st.number_input("Invoice Quantity", min_value=0, value=120)
        invoice_dollars = st.number_input("Invoice Dollars ($)", min_value=0.0, value=9500.0, step=50.0)
        freight_cost = st.number_input("Freight Cost ($)", min_value=0.0, value=150.0, step=5.0)
        total_brands = st.number_input("Total Brands on PO", min_value=1, value=3)

    with col2:
        total_item_quantity = st.number_input("Total Item Quantity (from PO)", min_value=0, value=118)
        total_item_dollars = st.number_input("Total Item Dollars (from PO, $)", min_value=0.0, value=4480.0, step=50.0)
        avg_receiving_delay = st.number_input("Avg Receiving Delay (days)", min_value=0.0, value=3.0, step=1.0)

    dollar_gap = abs(invoice_dollars - total_item_dollars)
    st.metric("Dollar Gap", f"${dollar_gap:,.2f}")

    if st.button("Check Invoice", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "Invoice_Quantity": invoice_quantity,
            "Invoice_Dollars": invoice_dollars,
            "Freight_Cost": freight_cost,
            "Total_Brands": total_brands,
            "Total_Item_Quantity": total_item_quantity,
            "Total_Item_Dollars": total_item_dollars,
            "Avg_Receiving_Delay": avg_receiving_delay,
            "Dollar_Gap": dollar_gap,
        }])

        X_scaled = scaler.transform(input_df)
        prediction = flag_model.predict(X_scaled)[0]
        probability = flag_model.predict_proba(X_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Flagged as risky — {probability:.0%} confidence")
            reasons = []
            if dollar_gap > 350:
                reasons.append(f"Dollar mismatch of ${dollar_gap:,.2f} exceeds the $350 threshold")
            if avg_receiving_delay > 10:
                reasons.append(f"Average receiving delay of {avg_receiving_delay:.0f} days exceeds 10 days")
            for r in reasons:
                st.write(f"- {r}")
        else:
            st.success(f"✅ Looks normal — {1 - probability:.0%} confidence")

    st.divider()
    st.caption("Sample data")
    st.dataframe(pd.DataFrame([
        {"Invoice_Quantity": 120, "Invoice_Dollars": 9500, "Freight_Cost": 150, "Total_Brands": 3,
         "Total_Item_Quantity": 118, "Total_Item_Dollars": 4480, "Avg_Receiving_Delay": 3, "Expected": "Flagged"},
        {"Invoice_Quantity": 50, "Invoice_Dollars": 2000, "Freight_Cost": 75, "Total_Brands": 1,
         "Total_Item_Quantity": 50, "Total_Item_Dollars": 1990, "Avg_Receiving_Delay": 2, "Expected": "Normal"},
    ]), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Predict expected freight cost")
    st.write("Enter the invoice dollar amount to get a predicted baseline freight cost.")

    dollars = st.number_input("Invoice Dollars ($)", min_value=0.0, value=3200.0, step=100.0, key="freight_dollars")

    if st.button("Predict Freight Cost", type="primary", use_container_width=True):
        pred = freight_model.predict(pd.DataFrame([{"Dollars": dollars}]))[0]
        st.metric("Predicted Freight Cost", f"${pred:,.2f}")
        st.caption(f"That's about {pred / dollars * 100:.1f}% of the invoice total.")

    st.divider()
    st.caption("Sample data")
    sample = pd.DataFrame({"Dollars": [1500, 3200, 4800, 6000]})
    sample["Predicted Freight Cost"] = freight_model.predict(sample[["Dollars"]]).round(2)
    st.dataframe(sample, use_container_width=True, hide_index=True)

st.divider()
st.caption("Models trained on real vendor invoice data · Random Forest classifier · Decision Tree regressor")