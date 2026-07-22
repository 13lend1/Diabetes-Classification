import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder,StandardScaler

def encode(df,col='Gender'):
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    return df

def scale(df):
    df_numeric=df.select_dtypes('number')
    scaler=StandardScaler()
    df_numeric=scaler.fit_transform(df_numeric)
    return df_numeric