from sklearn.neural_network import MLPClassifier
import optuna
from .model import Agent
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler,OneHotEncoder
MODEL=MLPClassifier()
class MLP(Agent):
    def __init__(self):
        super().__init__(MODEL)
        
    def scaling(self, X: pd.DataFrame, test: pd.DataFrame, flag_col: str = "flag"):
        scaler = StandardScaler().set_output(transform="pandas")
        ohe = OneHotEncoder(sparse_output=False, drop="first").set_output(transform="pandas")

        # separate categorical columns
        X_cat = X[['gender', flag_col]]
        X_num = X.drop(['gender', flag_col], axis=1)

        test_cat = test[['gender', flag_col]]
        test_num = test.drop(['gender', flag_col], axis=1)

        # fit scaler on train only, apply to both
        X_num = scaler.fit_transform(X_num)
        test_num = scaler.transform(test_num)

        # fit OHE on train only, apply to both
        X_cat_enc = ohe.fit_transform(X_cat)
        test_cat_enc = ohe.transform(test_cat)

        X_final = pd.concat([X_num, X_cat_enc], axis=1)
        test_final = pd.concat([test_num, test_cat_enc], axis=1)

        return X_final.reset_index(drop=True), test_final.reset_index(drop=True)
            
    def hyperparameter_tuning(self,X:np.ndarray,y:np.ndarray,trials:int)->dict:
        def objective(trial):
            params = {
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes",
                    [
                        (32,),
                        (64,),
                        (128,),
                        (64, 32),
                        (128, 64),
                        (128, 64, 32)
                    ]
                ),

                "activation": trial.suggest_categorical(
                    "activation",
                    ["relu", "tanh", "logistic"]
                ),

                "solver": trial.suggest_categorical(
                    "solver",
                    ["adam", "sgd"]
                ),

                "alpha": trial.suggest_float(
                    "alpha",
                    1e-6,
                    1e-2,
                    log=True
                ),

                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init",
                    1e-5,
                    1e-2,
                    log=True
                ),

                "batch_size": trial.suggest_categorical(
                    "batch_size",
                    [32, 64, 128, 256]
                ),

                "max_iter": trial.suggest_int(
                    "max_iter",
                    200,
                    1000,
                    step=100
                ),

                "early_stopping": True,

                "random_state": 42
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
    def save_model(model:MLPClassifier,path:str)->None:
        joblib.dump(model,path)