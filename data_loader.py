import pandas as pd
import os

def load_csv(dataset_path: str, filename: str = "data.csv") -> pd.DataFrame:
    """
    Carga dataset CSV desde un path local.
    """
    full_path = os.path.join(dataset_path, filename)
    return pd.read_csv(full_path)