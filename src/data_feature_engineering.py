import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from data_preprocessing import scale,encode
from sklearn.cluster import KMeans  
from sklearn.feature_selection import mutual_info_classif

df=pd.read_csv("data/clean/Diabetes Classifier Clean.csv")
y=df['Diagnosis']
X=df.drop(['Diagnosis'],axis=1)



X=encode(X,'Gender')
gender=X['Gender']

X.drop(['Gender'],axis=1)


eps = 1e-8

def features(X):
    X['Lipids']=X['HDL']/(X['LDL']+eps)
    X['Age_x_BMI']=X['Age']*X['BMI']
    X['Lipids_x_BMI']=X['BMI']/(X['Lipids']+eps)
    X['HDL_x_LDL']=X['HDL']*X['LDL']
    X['Age/HDL']=X['Age']/(X['HDL']+eps)
    X['Age/LDL']=X['Age']/(X['LDL']+eps)
    X['BMI/HDL']=X['BMI']/(X['HDL']+eps)
    X['BMI/LDL']=X['BMI']/(X['LDL']+eps)
    X['BMI/HDL+LDL']=X['BMI']/(X['LDL']+X['HDL']+eps)
    
    X_scaled=scale(X)

    kmeans=KMeans(n_clusters=6,n_init=10,random_state=42)
    kmeans.fit(X_scaled)
    # X['cluster_labels']=kmeans.labels_
    X['Gender']=gender

    distance=kmeans.transform(X_scaled)

    for i in range (distance.shape[1]):
        X[f'Cluster {i}']=distance[:,i]
    
    return X

X=features(X)

features= X.columns

mi_scores=mutual_info_classif(X,y,random_state=42,n_jobs=-1)

mi_scores_series = pd.Series(mi_scores, index=features).sort_values(ascending=False)

print(mi_scores_series)     

plt.figure(figsize=(10, 6))
mi_scores_series.plot(kind='barh', color='teal')
plt.title('Mutual Information Scores per Feature after FE')
plt.xlabel('Mutual Information Score')
plt.ylabel('Features')
plt.gca().invert_yaxis() # Top-down display
plt.tight_layout()
plt.show()