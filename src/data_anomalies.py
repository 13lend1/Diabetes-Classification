import pandas as pd
import numpy as np 
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

RANDOM_STATE=42
def outliersCheck(df: pd.DataFrame) -> set:
    df = df.copy()
    features = df.copy()  
    index = set()

    iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
    df['iso_anomalies'] = iso.fit_predict(features)

    iso_anomalies = df[df['iso_anomalies'] == -1]
    print("ISOLATION FOREST:")
    print(iso_anomalies.index)
    print(iso_anomalies.shape)
    index.update(iso_anomalies.index) 

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    df['lof_anomalies'] = lof.fit_predict(features)
    lof_anomalies_idx = df[df['lof_anomalies'] == -1].index
    print("LOF:")
    print(df.loc[lof_anomalies_idx, 'lof_anomalies'])
    print(lof_anomalies_idx.shape)
    print()

    index.update(lof_anomalies_idx)  

    return iso_anomalies.index,lof_anomalies_idx

