import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from dataclasses import dataclass
from sklearn.base import clone
import pandas as pd


@dataclass
class ModelMetrics:
    accuracy:np.ndarray
    precision:np.ndarray
    f1:np.ndarray
    recall: np.ndarray
    roc_auc:np.ndarray
    def __init__(self):
        pass
    
    def __init__(self, metrics: dict):
        self.accuracy = metrics['test_accuracy']
        self.precision = metrics['test_precision_macro']
        self.f1 = metrics['test_f1_macro']
        self.recall = metrics['test_recall_macro']
        self.roc_auc=metrics['test_roc_auc']
        
    def measure(self,y_pred:np.ndarray,y_real:np.ndarray):
        self.accuracy=accuracy_score(y_real,y_pred)
        self.precision=precision_score(y_real,y_pred)
        self.f1=f1_score(y_real,y_pred)
        self.recall=recall_score(y_real,y_pred)
        self.roc_auc=roc_auc_score(y_real,y_pred)

    def out(self):
        print(f"Accuracy: {self.accuracy.mean()}")
        print(f"Precision: {self.precision.mean()}")
        print(f"F1: {self.f1.mean()}")
        print(f"Recall: {self.recall.mean()}")
        print(f"ROC-AUC: {self.roc_auc.mean()}")
                                
    
    

class Agent:
    def __init__(self,model:BaseEstimator):
        self.model=model
        self.parameters=dict
        self.X=pd.DataFrame
        self.y=pd.Series
        
    def set_params(self,params:dict)->None:
        self.model.set_params(**params)
        
    def train(self,X:pd.DataFrame,y:pd.Series)->None:
        self.model.fit(X,y)
    
    def predict(self,x:pd.DataFrame)->np.ndarray:
        return self.model.predict(x)
    def predict_proba(self,X:pd.DataFrame)->np.ndarray:
        return self.model.predict_proba(X)

    def oof(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        name: str,
        k: int = 5,
        random_state: int = 42,
    ):

        skf = StratifiedKFold(
            n_splits=k,
            shuffle=True,
            random_state=random_state
        )

        oof_preds = np.zeros(len(X_train))
        test_preds = np.zeros((len(X_test), k))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):

            X_tr = X_train.iloc[train_idx].copy()
            X_val = X_train.iloc[val_idx].copy()

            y_tr = y_train.iloc[train_idx]

            # Scale only KNN and MLP
            if name in ["knn", "mlp"]:

                gender_tr = X_tr["gender"]
                gender_val = X_val["gender"]
                gender_test = X_test["gender"]

                X_tr = X_tr.drop(columns="gender")
                X_val = X_val.drop(columns="gender")
                X_te = X_test.drop(columns="gender")

                scaler = StandardScaler()

                X_tr = pd.DataFrame(
                    scaler.fit_transform(X_tr),
                    columns=X_tr.columns,
                    index=X_tr.index,
                )

                X_val = pd.DataFrame(
                    scaler.transform(X_val),
                    columns=X_val.columns,
                    index=X_val.index,
                )

                X_te = pd.DataFrame(
                    scaler.transform(X_te),
                    columns=X_te.columns,
                    index=X_test.index,
                )

                X_tr["gender"] = gender_tr
                X_val["gender"] = gender_val
                X_te["gender"] = gender_test

            else:
                X_te = X_test.copy()

            model = clone(self.model)

            model.fit(X_tr, y_tr)

            # OOF prediction
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

            # Test prediction
            test_preds[:, fold] = model.predict_proba(X_te)[:, 1]

            print(f"{name} fold {fold + 1}/{k} done")

        return (
            pd.DataFrame({f"{name}_oof": oof_preds}),
            pd.DataFrame({f"{name}_test": test_preds.mean(axis=1)})
        )
    def evaluation(self,X:np.ndarray,y:np.ndarray,k:int=5)->ModelMetrics:
        skf=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
        scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro','roc_auc']
        results=cross_validate(self.model,X,y,cv=skf,scoring=scoring_metrics)
        return ModelMetrics(results)


            
