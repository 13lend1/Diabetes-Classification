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
        


    def oof(self, X: pd.DataFrame, y: pd.Series, name: str, k: int = 5, random_state: int = 42):

        skf = StratifiedKFold(
            n_splits=k,
            shuffle=True,
            random_state=random_state
        )

        oof_preds = np.zeros(len(X))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()

            y_train = y.iloc[train_idx]

            # Scale only KNN and MLP
            if name in ["knn", "mlp"]:
                scaler = StandardScaler()

                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            model = clone(self.model)

            model.fit(X_train, y_train)

            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

            print(f"{name} fold {fold+1}/{k} done")

        return pd.DataFrame({
            f"{name}_oof": oof_preds
        })
    def evaluation(self,X:np.ndarray,y:np.ndarray,k:int=5)->ModelMetrics:
        skf=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
        scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro','roc_auc']
        results=cross_validate(self.model,X,y,cv=skf,scoring=scoring_metrics)
        return ModelMetrics(results)


            
