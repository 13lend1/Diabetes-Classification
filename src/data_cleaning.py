import pandas as pd
import numpy as np


def uncapitalize(df:pd.DataFrame)->pd.DataFrame:
    df.columns=df.columns.str.lower()
    return df

def drop_duplicates(df:pd.DataFrame)->pd.DataFrame:
    df.drop_duplicates(inplace=True)
    return df

#fixing small 'f' gender issue
def gender_fix(df:pd.DataFrame,col='gender')->pd.DataFrame:
    df[col]=df[col].apply(lambda x: 'F' if x=='f' else x)
    return df

def drop_col(df:pd.DataFrame,col='unnamed: 0')->pd.DataFrame:
    df.drop([col],axis=1,inplace=True)
    return df

