import pandas as pd
import numpy as np

df=pd.read_csv("data/raw/blood_cell_anomaly_detection.csv")

for col in df.columns:
    print(col,end=' ')
    print(df[col].dtype)
    print("----------------")
