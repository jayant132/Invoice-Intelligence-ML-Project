import os
import sqlite3
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(db_path):
    """
    Load data from SQLite database.
    """

    conn = sqlite3.connect(db_path)

    query = """
    WITH purchase_agg AS (
        SELECT
            p.PO_Number,
            COUNT(DISTINCT p.Brand) AS Total_Brands,
            SUM(p.Item_Quantity) AS Total_Item_Quantity,
            SUM(p.Dollars) AS Total_Item_Dollars,
            AVG(JULIANDAY(p.Receiving_Date) - JULIANDAY(p.PO_Date)) AS Avg_Receiving_Delay
        FROM Purchases p
        GROUP BY p.PO_Number
    )

    SELECT
        vi.Quantity AS Invoice_Quantity,
        vi.Dollars AS Invoice_Dollars,
        vi.Freight_Cost,
        JULIANDAY(vi.Invoice_Date) - JULIANDAY(vi.PO_Date) AS Days_PO_To_Invoice,
        JULIANDAY(vi.Pay_Date) - JULIANDAY(vi.Invoice_Date) AS Days_To_Pay,
        pa.Total_Brands,
        pa.Total_Item_Quantity,
        pa.Total_Item_Dollars,
        pa.Avg_Receiving_Delay
    FROM Vendor_Invoice vi
    JOIN purchase_agg pa
    ON vi.PO_Number = pa.PO_Number;
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    return df


def create_invoice_risk_label(row):
    """
    Create fraud label.
    """

    if abs(row["Invoice_Dollars"] - row["Total_Item_Dollars"]) > 350:
        return 1

    if row["Avg_Receiving_Delay"] > 10:
        return 1

    return 0


def preprocess_data(df):
    """
    Feature Engineering
    """

    df["Dollar_Gap"] = (df["Invoice_Dollars"] - df["Total_Item_Dollars"]).abs()

    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)

    return df


def prepare_data(df):
    """
    Select features and target.
    """

    X = df[
        [
            "Invoice_Quantity",
            "Invoice_Dollars",
            "Freight_Cost",
            "Total_Brands",
            "Total_Item_Quantity",
            "Total_Item_Dollars",
            "Avg_Receiving_Delay",
            "Dollar_Gap",
        ]
    ]

    y = df["flag_invoice"]

    return X, y


def split_and_scale(X, y):
    """
    Train-Test Split
    Scaling
    Save scaler using Joblib
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    print("Scaler saved successfully at models/scaler.pkl")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
    )