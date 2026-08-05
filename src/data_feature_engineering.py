import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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

    # Save gender separately
    gender_train = X_train[['gender']]
    gender_test = X_test[['gender']]

    X_train = X_train.drop(columns='gender')
    X_test = X_test.drop(columns='gender')

    # ------------------------
    # Scaling
    # ------------------------
    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------
    # PCA
    # ------------------------
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    for i in range(X_train_pca.shape[1]):
        X_train[f'PCA_{i+1}'] = X_train_pca[:, i]
        X_test[f'PCA_{i+1}'] = X_test_pca[:, i]

    # ------------------------
    # KMeans
    # ------------------------
    kmeans = KMeans(
        n_clusters=6,
        n_init=30,
        random_state=42
    )

    kmeans.fit(X_train_scaled)

    X_train['cluster_labels'] = kmeans.labels_
    X_test['cluster_labels'] = kmeans.predict(X_test_scaled)

    train_dist = kmeans.transform(X_train_scaled)
    test_dist = kmeans.transform(X_test_scaled)

    for i in range(train_dist.shape[1]):
        X_train[f'Cluster_{i}'] = train_dist[:, i]
        X_test[f'Cluster_{i}'] = test_dist[:, i]

    # Add gender back
    X_train = pd.concat([X_train, gender_train], axis=1)
    X_test = pd.concat([X_test, gender_test], axis=1)

    return X_train, X_test