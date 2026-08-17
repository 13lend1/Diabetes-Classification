# Diabetes Classifier Machine Learning End-to-End Project 
My first end-to-end Machine Learning project that predicts diabetes based on blood panel results!
## About the project 
    - The project is done for gaining experience and training my personal ML skills.
    - The dataset used in this project is found in [https://www.kaggle.com/datasets/simaanjali/diabetes-classification-dataset](https://www.kaggle.com/datasets/simaanjali/diabetes-classification-dataset)
    -This project uses a stacking ensemble to classify.
    - The pipeline of this project is data cleaning -> data preprocessing and feature engineering 
    -> base models training -> base models OOF predictions -> meta model training on OOF data -> meta model prediction
    ->Base models used are RandomForsetClassifier, XGBClassifier, AdaBoostClassifier and MLPClassifier.
    ->As a Meta Model was used Logistic Regression.

# How to use
## Installation and setup 
    1.Clone the repository 
    '''git clone https://github.com/13lend1/Diabetes-Classification'''
    2.Create and Activate a virtual environment
    '''uv innit'''
    3.Install the dependencies 
    '''uv pip install -r requirements.txt'''

## Requiremtens
    Here: [requirements.txt]('requirements.txt')

## Usage 
    The model is deployed using Gradio, which provides a web interface.
    To use the model locally for prediction run :
    '''deployment\app.py'''
    After you run the file, in your terminal will appear a link, open the link on your default browser and enter the required data!

# Pipeline
    I highly reccomend having a look at the notebooks folder, there you can see the whole pipeline where I've done [data investigation](notebooks\0_data_investigate.ipynb), [data cleaning](notebooks\1_data_cleaning.ipynb), [data visualization](notebooks\2_data_visualization.ipynb), [anomalies](notebooks\3_anomalies.ipynb), [feature engineering](notebooks\4_feature_engineering.ipynb), [model fine tuning](notebooks\5_model.ipynb) and [stacking ensmble](notebooks\6_stacking_ensemble.ipynb)

# What needs improvement
    If you check out the [anomalies](notebooks\3_anomalies.ipynb)notebook you'll find an issue i ran into dealing with anomalies where neither dropping or imputing them wouldn't really help!
# How to Contribute 
    You're more then welcome to contribute on this project!
    1.Fork the repository
    2.Create a new branch 
    3.Make changes 
    4.Commit and push the changes on your branch 
    5.Open a pull request  
