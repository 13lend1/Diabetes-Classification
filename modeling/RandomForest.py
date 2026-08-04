from sklearn.ensemble import RandomForestClassifier
import optuna
from .model import Agent
import pandas as pd
import numpy as np
import joblib

MODEL=RandomForestClassifier()
class RandomForest(Agent):
    def __init__(self):
        super().__init__(MODEL)
    def hyperparameter_tuning(self,X:np.ndarray,y:np.ndarray,trials:int)->dict:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),

                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),

                "bootstrap": trial.suggest_categorical(
                    "bootstrap", [True, False]
                ),

                "criterion": trial.suggest_categorical(
                    "criterion", ["gini", "entropy", "log_loss"]
                ),

                "class_weight": trial.suggest_categorical(
                    "class_weight", [None, "balanced", "balanced_subsample"]
                ),

                "n_jobs": -1,
                "random_state": 42,
            }

            self.model.set_params(**params)

            score = self.evaluation(X, y, 5)

            print(params)
            score.out()

            return score.accuracy.mean()


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
    def save_model(model:RandomForestClassifier,path:str)->None:
        joblib.dump(model,path)
    