# Diabetes Classifier Machine Learning End-to-End Project

My first end-to-end Machine Learning project that predicts diabetes based on blood panel results!

`Disclaimer`: This project is for educational purposes only and is not intended for medical use. It has not been clinically validated and should not be used to make real health decisions. Consult a qualified healthcare professional for medical advice.

## About the project

- The project is done for gaining experience and training my personal ML skills.
- The dataset used in this project is found on [Kaggle](https://www.kaggle.com/datasets/simaanjali/diabetes-classification-dataset).
- This project uses a stacking ensemble to classify.
- The pipeline of this project is: data cleaning -> data preprocessing and feature engineering -> base models training -> base models Out-Of-Fold (OOF) predictions -> meta model training on OOF data -> meta model prediction.
- Base models used are `RandomForestClassifier`, `XGBClassifier`, `AdaBoostClassifier`, and `MLPClassifier`.
- The meta model used is Logistic Regression.

## How to use

### Installation and setup

1. Clone the repository
   ```bash
   git clone https://github.com/13lend1/Diabetes-Classification
   ```
2. Create and activate a virtual environment
   ```bash
   uv venv
   ```
3. Install the dependencies
   ```bash
   uv pip install -r requirements.txt
   ```

### Requirements

See [requirements.txt](requirements.txt) for the full list of dependencies.

### Usage

The model is deployed using Gradio, which provides a web interface.
To use the model locally for prediction, run:
```bash
python deployment/app.py
```
After you run the file, a link will appear in your terminal. Open it in your default browser and enter the required data!

## Pipeline

> **Note:** The notebooks were developed and run using an Anaconda environment/kernel, separate from the Gradio deployment `.venv`. To run them yourself, set up an Anaconda environment and select it as the Jupyter kernel before opening the notebooks.

I highly recommend having a look at the `notebooks` folder, where you can see the whole pipeline:

- [Data investigation](notebooks/0_data_investigate.ipynb)
- [Data cleaning](notebooks/1_data_cleaning.ipynb)
- [Data visualization](notebooks/2_data_visualization.ipynb)
- [Anomalies](notebooks/3_anomalies.ipynb)
- [Feature engineering](notebooks/4_feature_engineering.ipynb)
- [Model fine tuning](notebooks/5_model.ipynb)
- [Stacking ensemble](notebooks/6_stacking_ensemble.ipynb)

## What needs improvement

If you check out the [anomalies](notebooks/3_anomalies.ipynb) notebook, you'll find an issue I ran into dealing with anomalies, where neither dropping nor imputing them really helped.

## How to Contribute

You're more than welcome to contribute to this project!

1. Fork the repository
2. Create a new branch
3. Make changes
4. Commit and push the changes on your branch
5. Open a pull request