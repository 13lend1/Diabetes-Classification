import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
from dataclasses import dataclass
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
    
    def out(self):
        print(f"Accuracy: {self.accuracy.mean()}")
        print(f"Precision: {self.precision.mean()}")
        print(f"F1: {self.f1.mean()}")
        print(f"Recall: {self.recall.mean()}")
                                
    
    

class Agent:
    def __init__(self,model:BaseEstimator):
        self.model=model
        self.parameters=dict
        self.X=pd.DataFrame
        self.y=pd.Series
        
    def set_params(self,params:dict)->None:
        self.model.set_params(params)
        
    def train(self,X:pd.DataFrame,y:pd.Series)->None:
        self.model.fit(X,y)
        
    def evaluation(self,X:np.ndarray,y:np.ndarray,k:int=5)->ModelMetrics:
        scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        results=cross_validate(self.model,X,y,cv=k,scoring=scoring_metrics)
        self.model.fit(X,y)
        return ModelMetrics(results)


            
