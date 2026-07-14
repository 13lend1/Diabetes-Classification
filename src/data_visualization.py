import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns 
pd.set_option('display.max_columns',None)
df=pd.read_csv("data/clean/Diabetes Classifier Clean.csv")

diabetics=(df['Diagnosis']==1).sum()
nonDiabetics=(df['Diagnosis']==0).sum()

classes=['Diabetics','Non Diabetics']
count=[diabetics,nonDiabetics]

plt.bar(classes,count,color='orange')
plt.title("Target Distribution")
plt.ylabel("Count")
plt.xlabel("Classes")
plt.show()

# We can see that the target is well distributed roughly 60/40

sns.histplot(df['Gender'],bins=2,shrink=0.8, edgecolor='black')
plt.show()

# Theres more data for male patients 

sns.histplot(df['Age'],bins=100,shrink=0.8, edgecolor='black',color='brown')
plt.show()
# Most of the patients of the dataset are around 30-40 or 50-60 years old


fig, ax = plt.subplots(figsize=(10,6))
bins = [10,12,14,16,18,18.5,20,22,24,24.9,26,28,30,32,34,36,38,40]  # finer near cutoffs

sns.histplot(df['BMI'], bins=bins, ax=ax)

for patch in ax.patches:
    x_start = patch.get_x()
    if x_start < 18.5:
        patch.set_facecolor('red')
    elif x_start < 24.9:
        patch.set_facecolor('green')
    else:
        patch.set_facecolor('red')

legend_patches = [
    mpatches.Patch(color='red', label='Unhealthy (<18.5 or ≥25)'),
    mpatches.Patch(color='green', label='Healthy (18.5–24.9)')
]
ax.legend(handles=legend_patches)
ax.set_title('BMI Distribution')
plt.show()

def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Healthy'
    else:
        return 'Overweight/Obese'

df['BMI_Category'] = df['BMI'].apply(bmi_category)

df['Diabetes_Label'] = df['Diagnosis'].map({0: 'No Diabetes', 1: 'Diabetes'})

order = ['Underweight','Healthy','Overweight/Obese']
sns.countplot(data=df, x='BMI_Category', hue='Diabetes_Label', order=order,
              palette={'No Diabetes':'lightgray', 'Diabetes':'crimson'})
plt.title('Diabetes Status by BMI Category')
plt.ylabel('Count')
plt.show()
print(df['BMI_Category'].value_counts())