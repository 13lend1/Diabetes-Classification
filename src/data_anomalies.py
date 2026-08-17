import pandas as pd
import numpy as np 
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import KNNImputer

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
    print(lof_anomalies_idx)

    index.update(lof_anomalies_idx)  

    return iso_anomalies.index,lof_anomalies_idx

def anomaly_imputer(df:pd.DataFrame)->pd.DataFrame:
    df = df.copy()  
    feature_cols = df.columns  

    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(df)
    df_sc = pd.DataFrame(df_scaled, columns=feature_cols, index=df.index)

    imputer = KNNImputer(n_neighbors=5)
    df_imputed = imputer.fit_transform(df_sc)
    df_imp = pd.DataFrame(df_imputed, columns=feature_cols, index=df.index)

    df_f = scaler.inverse_transform(df_imputed)
    df_final=pd.DataFrame(df_f,columns=feature_cols,index=df.index)
    return df_final

