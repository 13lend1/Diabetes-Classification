import pandas as pd
from src.data_preprocessing import scale
from sklearn.cluster import KMeans

def features(X:pd.DataFrame)->pd.DataFrame:
    X=X.copy()
    eps = 1e-8
    X['lipids']=X['hdl']/(X['ldl']+eps)
    X['Age_x_BMI']=X['age']*X['bmi']
    X['HDL_x_LDL']=X['hdl']*X['ldl']
    X['BMI/LDL']=X['bmi']/(X['ldl']+eps)
    X['BMI/HDL+LDL']=X['bmi']/(X['ldl']+X['hdl']+eps)
    X['bun_x_cr']=X['cr']*X['bun']
    X['chol/ldl']=X['chol']/(X['ldl']+eps)
    
    gender=X[['gender']]
    X=X.drop('gender',axis=1)
    X_scaled=scale(X)

    kmeans=KMeans(n_clusters=6,n_init=10,random_state=42)
    kmeans.fit(X_scaled)
    X['cluster_labels']=kmeans.labels_

    distance=kmeans.transform(X_scaled)

    for i in range (distance.shape[1]):
        X[f'Cluster {i}']=distance[:,i]
    
    X=pd.concat([X,gender],axis=1)
    return X
