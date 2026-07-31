from sklearn.svm import SVC
import optuna
from .model import Agent
import pandas as pd
import numpy as np
import joblib

MODEL=SVC(probability=True)
class SVC(Agent):
    def __init__(self):
        super().__init__(MODEL)
        
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, trials: int) -> dict:
        def objective(trial):
            params = {
                "C": trial.suggest_float("C", 1e-3, 1e3, log=True),

                "kernel": trial.suggest_categorical(
                    "kernel",
                    ["linear", "poly", "rbf", "sigmoid"]
                ),

                "gamma": trial.suggest_categorical(
                    "gamma",
                    ["scale", "auto"]
                ),

                "degree": trial.suggest_int(
                    "degree", 2, 5
                ),

                "coef0": trial.suggest_float(
                    "coef0", 0.0, 2.0
                ),

                "shrinking": trial.suggest_categorical(
                    "shrinking",
                    [True, False]
                ),

                "tol": trial.suggest_float(
                    "tol", 1e-5, 1e-2, log=True
                ),

                "class_weight": trial.suggest_categorical(
                    "class_weight",
                    [None, "balanced"]
                ),

                "probability": True,
                "random_state": 42,
            }

            self.model.set_params(**params)

            score = self.evaluation(X, y, 5)

            print(params)
            score.out()

            return score.roc_auc.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials)

        print("Optimization complete!")
        print(f"Best Trial ROC AUC: {study.best_value:.4f}")

        print("Best Hyperparameters Found:")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")

        return study.best_params
                    
    def feature_importance(self,cols)->None:
        importance_df = pd.DataFrame({
        'Feature': cols,
        'Importance': self.model.feature_importances_
            }).sort_values(by='Importance', ascending=False)  
        print(importance_df)    
            
    @staticmethod
    def save_model(model:SVC,path:str)->None:
        joblib.dump(model,path)