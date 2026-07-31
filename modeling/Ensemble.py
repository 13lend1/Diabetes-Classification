import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from src.data_preprocessing import scaling
from src.data_feature_engineering import features
from sklearn.model_selection import train_test_split
import joblib
from modeling.model import Agent
from sklearn.metrics import f1_score,recall_score,precision_score,balanced_accuracy_score,roc_auc_score,accuracy_score

MODEL=LogisticRegression()

class LR(Agent):
    def __init__(self):
        super().__init__(MODEL)
        
                    
    def feature_importance(self,cols)->None:
        feature_cols=cols
        coef_df = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': self.model.coef_[0] 
        }).sort_values('coefficient', key=abs, ascending=False)
        print(coef_df)
            
    @staticmethod
    def save_model(model:LogisticRegression,path:str)->None:
        joblib.dump(model,path)
        
class StackingEnsemble:
    def __init__(self, base_agents: list, meta_model, df,oof_folds:int, target="diagnosis",scale_resistant_models=None):
        self.base_agents = base_agents
        self.meta_model = meta_model
        self.oof_folds=oof_folds
        self.meta_feature_names = None
        self.scale_resistant_models = scale_resistant_models or ["XGBoost", "RandomForest"]
        self.df = df
        self.y = self.df[target]
        self.X = self.df.drop(target, axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,test_size=0.2, random_state=42
        )
        
        self.X_train,self.X_test=features(self.X_train,self.X_test)
        self.X_train_scaled,self.X_test_scaled=scaling(self.X_train,self.X_test)
        self.meta_train = None
        self.meta_test = None
        
    def build_meta_features(self, k=10, random_state=42):
        train_parts, test_parts = [], []
        for agent in self.base_agents:
            oof_df, test_df = agent.oof(
                self.X_train, self.y_train, self.X_test,
                name=agent.__class__.__name__, k=k, random_state=random_state
            )
            oof_df = oof_df.rename(columns={f"{agent.__class__.__name__}_oof": f"{agent.__class__.__name__}_pred"})
            test_df = test_df.rename(columns={f"{agent.__class__.__name__}_test": f"{agent.__class__.__name__}_pred"})

            train_parts.append(oof_df)
            test_parts.append(test_df)

        train_parts.append(self.X_train_scaled.reset_index(drop=True))
        test_parts.append(self.X_test_scaled.reset_index(drop=True))

        self.meta_train = pd.concat(train_parts, axis=1)
        self.meta_train.index = self.X_train.index

        self.meta_test = pd.concat(test_parts, axis=1)
        self.meta_test.index = self.X_test.index

        return self.meta_train, self.meta_test

    def fit_base_models(self):
        # final models used for deployment inference — full X_train, no OOF
        for agent in self.base_agents:
            X_input = self.X_train if agent.__class__.__name__ in self.scale_resistant_models else self.X_train
            agent.train(X_input)

    def fit_meta_model(self):
        if self.meta_train is None:
            self.build_meta_features()
        self.meta_feature_names = list(self.meta_train.columns)
        self.meta_model.train(self.meta_train, self.y_train)
        print(self.meta_train)
        print(self.y_train)
        print(self.meta_train.shape)
        print(self.y_train.shape)

    def predict_proba(self):
        if self.meta_test is None:
            self.build_meta_features()
            print(f"{self.meta_test.columns} $and$ {self.meta_feature_names}")
        assert list(self.meta_test.columns) == self.meta_feature_names, "Column mismatch!"
        return self.meta_model.predict_proba(self.meta_test)
    
    def predict(self):
        if self.meta_test is None:
            self.build_meta_features()
        print(f"{self.meta_test.columns} $and$ {self.meta_feature_names}")
        assert list(self.meta_test.columns) == self.meta_feature_names, "Column mismatch!"
        return self.meta_model.predict(self.meta_test)
    
    def measure(self):
        metrics = {}
        self.fit_meta_model()
        y_pred=self.predict()
        y_proba=self.predict_proba()
        y_real=self.y_test
        metrics['accuracy'] = accuracy_score(y_real, y_pred)
        metrics['balanced']=balanced_accuracy_score(y_real,y_pred)
        metrics['precision'] = precision_score(y_real, y_pred)
        metrics['f1'] = f1_score(y_real, y_pred)
        metrics['recall'] = recall_score(y_real, y_pred)
        metrics['roc_auc'] = roc_auc_score(y_real, y_proba[:,1])
        print(metrics)