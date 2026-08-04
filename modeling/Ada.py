from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import optuna
from .model import Agent
import pandas as pd
import numpy as np
import joblib

MODEL=AdaBoostClassifier()
class Ada(Agent):
    def __init__(self):
        super().__init__(MODEL)
        
    def hyperparameter_tuning(self, X, y, trials: int) -> dict:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
                "random_state": 42
            }

            model = AdaBoostClassifier(**params)

            score = cross_val_score(
                model,
                X,
                y,
                cv=5,
                scoring="balanced_accuracy",
                n_jobs=-1
            ).mean()

            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials)

        return study.best_params
                    
    def feature_importance(self,cols)->None:
        importance_df = pd.DataFrame({
        'Feature': cols,
        'Importance': self.model.feature_importances_
            }).sort_values(by='Importance', ascending=False)  
        print(importance_df)    
            
    @staticmethod
    def save_model(model:AdaBoostClassifier,path:str)->None:
        joblib.dump(model,path)