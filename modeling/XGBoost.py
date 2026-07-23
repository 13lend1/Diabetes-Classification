from xgboost import XGBClassifier
import optuna
from .model import Agent
import pandas as pd
import numpy as np
import joblib
from typing import Dict,Any 
params={'n_estimators': 10500, 'max_depth': 14, 'learning_rate': 0.012882893680375776, 'gamma': 1.1351700414001096e-07, 'min_child_weight': 6, 'reg_alpha': 3.9476135030729944, 'reg_lambda': 1.695388534303708e-05, 'subsample': 0.20027376138250352, 'colsample_bytree': 0.6138826479887023, 'tree_method': 'hist', 'n_jobs': -1}
MODEL=XGBClassifier(**params)
class XGBoost(Agent):
    def __init__(self):
        
        super().__init__(MODEL)
    def hyperparameter_tuning(self,X:np.ndarray,y:np.ndarray):
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 3000, 12000, step=100),
                "max_depth": trial.suggest_int("max_depth", 6, 20),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                
                # Regularization (Structural Control)
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                
                # Sampling (Stochastic Control)
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                
                "tree_method": "hist",
                "n_jobs": -1}
            xgb=XGBClassifier(**params)
            score=self.train_eval2(xgb,X,y,5)
            print(params)
            score.out()
            return score
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        
        print("Optimization complete!")
        print(f"Best Trial Accuracy Score: {study.best_value:.4f}")
        print("Best Hyperparameters Found:")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")
            
            
    def prediction(self,x:pd.DataFrame)->np.ndarray:
        return self.model.predict(x)
    
    @staticmethod
    def save_model(model:XGBClassifier,path:str)->None:
        joblib.dump(model,path)