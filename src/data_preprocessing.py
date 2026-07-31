import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder

def encode(df,col='gender'):
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    return df

def scale(df):
    df_numeric=df.select_dtypes('number')
    scaler=StandardScaler()
    df_numeric=scaler.fit_transform(df_numeric)
    return df_numeric

def scaling(X: pd.DataFrame, test: pd.DataFrame, flag_col: str = "anomaly_score"):
    scaler = StandardScaler().set_output(transform="pandas")
    ohe = OneHotEncoder(sparse_output=False, drop="first").set_output(transform="pandas")

    # separate categorical columns
    X_cat = X[['gender', flag_col]]
    X_num = X.drop(['gender', flag_col], axis=1)

    test_cat = test[['gender', flag_col]]
    test_num = test.drop(['gender', flag_col], axis=1)

    # fit scaler on train only, apply to both
    X_num = scaler.fit_transform(X_num)
    test_num = scaler.transform(test_num)

    # fit OHE on train only, apply to both
    X_cat_enc = ohe.fit_transform(X_cat)
    test_cat_enc = ohe.transform(test_cat)

    X_final = pd.concat([X_num, X_cat_enc], axis=1)
    test_final = pd.concat([test_num, test_cat_enc], axis=1)

    return X_final.reset_index(drop=True), test_final.reset_index(drop=True)

# def scaling(X: pd.DataFrame,flag_col: str = "anomaly_score"):
#     scaler = StandardScaler().set_output(transform="pandas")
#     ohe = OneHotEncoder(sparse_output=False, drop="first").set_output(transform="pandas")

#     # separate categorical columns
#     X_cat = X[['gender', flag_col]]
#     X_num = X.drop(['gender', flag_col], axis=1)


#     # fit scaler on train only, apply to both
#     X_num = scaler.fit_transform(X_num)

#     # fit OHE on train only, apply to both
#     X_cat_enc = ohe.fit_transform(X_cat)

#     X_final = pd.concat([X_num, X_cat_enc], axis=1)
#     return X_final.reset_index(drop=True)