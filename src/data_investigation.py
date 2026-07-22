import pandas as pd
import numpy as np

def featureType(df:pd.DataFrame):
    numerical=df.select_dtypes(include=[np.number]).columns
    categorical=df.select_dtypes(include='object').columns
    return numerical,categorical

def cardinality(categorical,df:pd.DataFrame):
    for col in categorical:
        print(df[col].unique())
        print(df[col].value_counts())
        

