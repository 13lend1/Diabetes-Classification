
import gradio as gr
import joblib
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np 
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.configuration import CLEAN_DATA,MODEL_PATH
import joblib 

CLEAN_DATA=CLEAN_DATA.lstrip("..\\")
@dataclass
class Config:
    MODEL_PATH:Path=Path(MODEL_PATH)
    EXPORT_PATH:Path=Path(CLEAN_DATA)
    


class DiabetesPredictor:
    def __init__(self,config:Config):
        self.config=config
        self._load_model()
        self._prepare_data()
        
    def _load_model(self):
        self.prediction_model=joblib.load(self.config.MODEL_PATH)
    
    def _prepare_data(self):
        self.df=pd.read_csv(self.config.EXPORT_PATH)
        
    def predict(self,age,gender,bmi,chol,tg,hdl,ldl,cr,bun):
        gender = 1 if gender == "M" else 0
        inputs = {
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "chol": chol,
            "tg": tg,
            "hdl": hdl,
            "ldl": ldl,
            "cr": cr,
            "bun": bun,
        }
        row=pd.DataFrame([inputs])
        return self.prediction_model.predict_single(row)
    
class DiabetesPredictorUI:
    
    def __init__(self,predictor:DiabetesPredictor,config:Config):
        self.predictor=predictor
        self.config=config
    
    def create_interface(self)->gr.Interface:
        inputs = [
            gr.Number(label="Age"),
            gr.Radio(["M", "F"], label="Gender"),
            gr.Number(label="BMI(mmol/L)"),
            gr.Number(label="Cholesterol(mmol/L)"),
            gr.Number(label="Triglycerides(mmol/L)"),
            gr.Number(label="HDL(mmol/L)"),
            gr.Number(label="LDL(mmol/L)"),
            gr.Number(label="Creatinine(µmol/L)"),
            gr.Number(label="BUN(mmol/L)"),
                ]    
        return gr.Interface(
            fn=self.predictor.predict,
            inputs=inputs,
            outputs="text",
            title="Diabetes Predictor",
            theme=gr.themes.Soft(),
        )    
        
        
def main():
    config = Config()
    predictor = DiabetesPredictor(config)
    ui = DiabetesPredictorUI(predictor, config)
    ui.create_interface().launch(share=True)


if __name__ == "__main__":
    main()