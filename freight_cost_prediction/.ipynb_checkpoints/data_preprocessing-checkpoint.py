import sqlite3
from sklearn.model_selection import train_test_split
import pandas as pd

def load_vendor_invoice_data(db_path: str):
    """
    Load Vendor Invoice Data from Sql lite Database
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM Vendor_Invoice"
    df = pd.read_sql_query(query,conn)
    conn.close()
    return df


def prepare_features(df: pd.DataFrame):
    """
    Select Features and Target Variable
    """
    X = df[["Dollars"]]
    Y = df["Freight_Cost"]
    return X,Y

def split_data(X,y, test_size = 0.2, random_state = 42):
    """
    Split Dataset into Train and Test sets
    
    """
    return train_test_split(
        X,y, test_size =test_size, random_state = random_state
    )