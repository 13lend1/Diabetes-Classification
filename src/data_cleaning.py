import pandas as pd
import numpy as np
from data_investigation import cardinality,featureType
from helperFunc import saveFile

PATH='data/clean/Diabetes Classifier Clean.csv'

df=pd.read_csv("data/raw/Diabetes Classification.csv")
df=df.copy()
print(df.shape)

# drop duplicates/if any
df=df.drop_duplicates()
print(df.shape)

#fixing small 'f' gender issue
df['Gender']=df['Gender'].apply(lambda x: 'F' if x=='f' else x)

num,cat=featureType(df)
cardinality(cat,df) 

df=df.drop(['Unnamed: 0'],axis=1)


saveFile(df,PATH)