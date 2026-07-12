import pandas as pd
import numpy as np

df=pd.read_csv("data/raw/Diabetes Classification.csv")
pd.set_option('display.max_columns',None)
print(df.info())
print(df.describe())
# testing if all the features belong to both of those categories
print("Shape: ",end='')
print(df.shape)

print("Columns: ",end='')
print(df.columns)

# checking for nulls 
def nullCheck(df):
    for col in df.columns:
        nulls=df[col].isna().sum()
        if(nulls>0):
            print(col,end=' ')
            print(" has",end=' ')
            print(nulls,end=' ')
            print("amount of nulls!")
        
nullCheck(df) #as in the datset description the dataset has no missing values   

# divide numerical and categorical features
def featureType(df):
    numerical=df.select_dtypes(include=[np.number]).columns
    categorical=df.select_dtypes(include='str').columns
    return numerical,categorical

numerical,categorical=featureType(df)
print("Categorical: "+str(categorical.size))
print("NUmeric: "+str(numerical.size))
print(numerical.size+categorical.size)


# Check for cardinality of categorical features
def cardinality(categorical,df):
    for col in categorical:
        print(df[col].unique())
        print(df[col].value_counts())
        
cardinality(categorical,df) #outputs the unique values and their amount

# Checking for target imbalance
target=df['Diagnosis']
print(target.value_counts(normalize=True)*100)