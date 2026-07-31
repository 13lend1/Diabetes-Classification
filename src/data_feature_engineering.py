import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

def features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_train = X_train.copy()
    X_test = X_test.copy()
    eps = 1e-8

    for X in (X_train, X_test):
        X['lipids'] = X['hdl'] / (X['ldl'] + eps)
        X['Age_x_BMI'] = X['age'] * X['bmi']
        X['HDL_x_LDL'] = X['hdl'] * X['ldl']
        X['BMI/LDL'] = X['bmi'] / (X['ldl'] + eps)
        X['BMI/HDL+LDL'] = X['bmi'] / (X['ldl'] + X['hdl'] + eps)
        X['bun_x_cr'] = X['cr'] * X['bun']
        X['chol/ldl'] = X['chol'] / (X['ldl'] + eps)

    gender_train = X_train[['gender']]
    gender_test = X_test[['gender']]
    X_train = X_train.drop('gender', axis=1)
    X_test = X_test.drop('gender', axis=1)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)   
    X_test_scaled = scaler.transform(X_test)         

    kmeans = KMeans(n_clusters=6, n_init=10, random_state=42)
    kmeans.fit(X_train_scaled)                         # fit on train only

    X_train['cluster_labels'] = kmeans.labels_
    X_test['cluster_labels'] = kmeans.predict(X_test_scaled)  # assign, don't refit

    train_dist = kmeans.transform(X_train_scaled)
    test_dist = kmeans.transform(X_test_scaled)

    for i in range(train_dist.shape[1]):
        X_train[f'Cluster {i}'] = train_dist[:, i]
        X_test[f'Cluster {i}'] = test_dist[:, i]

    X_train = pd.concat([X_train, gender_train], axis=1)
    X_test = pd.concat([X_test, gender_test], axis=1)

    return X_train, X_test





