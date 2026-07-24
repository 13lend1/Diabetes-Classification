from xgboost import XGBClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
# from lightgbm import LGBMClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.svm import SVC
import optuna
from .model import Agent
import pandas as pd
import numpy as np
import joblib

params={'n_estimators': 9700, 'max_depth': 15, 'learning_rate': 0.038843676013196854, 'gamma': 0.7567769478114804, 'min_child_weight': 1, 'reg_alpha': 3.3882727085705906e-08, 'reg_lambda': 5.986317777889251, 'subsample': 0.7059962897911474, 'colsample_bytree': 0.6383736163346578, 'tree_method': 'hist', 'n_jobs': -1}

MODEL=RandomForestClassifier(   n_estimators=500,          # more trees = more stable, diminishing returns past ~500 on 5k rows
    max_depth=8,               # keeps trees from overfitting on a small dataset
    min_samples_split=10,      # requires more evidence before splitting — good for small data
    min_samples_leaf=4,        # smooths predictions, reduces variance
    max_features='sqrt',       # standard for classification, good diversity across trees
    class_weight='balanced',   # handles imbalance without manual sample_weight plumbing
    random_state=42,
    n_jobs=-1)
class XGBoost(Agent):
    def __init__(self):
        
        super().__init__(MODEL)
    
    def hyperparameter_tuning(self,X:np.ndarray,y:np.ndarray,trials:int)->dict:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 3000, 12000, step=100),
                "max_depth": trial.suggest_int("max_depth", 6, 20),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                
                "tree_method": "hist",
                "n_jobs": -1}
            self.model.set_params(**params)
            score=self.evaluation(X,y,5)
            print(params)
            score.out()
            return score.recall.mean()
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials)
        
        print("Optimization complete!")
        print(f"Best Trial Accuracy Score: {study.best_value:.4f}")
        print("Best Hyperparameters Found:")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")
        
        return study.best_params
            
    def predict(self,x:pd.DataFrame)->np.ndarray:
        return self.model.predict(x)
    
    def feature_importance(self,cols)->None:
        importance_df = pd.DataFrame({
        'Feature': cols,
        'Importance': self.model.feature_importances_
         }).sort_values(by='Importance', ascending=False)  
        print(importance_df)    
        
    @staticmethod
    def save_model(model:XGBClassifier,path:str)->None:
        joblib.dump(model,path)