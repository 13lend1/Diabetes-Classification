import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns 
pd.set_option('display.max_columns',None)
df=pd.read_csv("data/clean/Diabetes Classifier Clean.csv")
df['Diabetes_Label'] = df['Diagnosis'].map({0: 'No Diabetes', 1: 'Diabetes'})
total = len(df)
def labeler(total,ax):
    for container in ax.containers:
        labels = [f'{v.get_height()/total*100:.1f}%' for v in container]
        ax.bar_label(container, labels=labels)

def grouped_labeler(ax, df, group_col, hue_order):
    totals = df[group_col].value_counts()
    for container in ax.containers:
        labels = [f'{bar.get_height()/totals[g]*100:.1f}%' 
                  for bar, g in zip(container, hue_order)]
        ax.bar_label(container, labels=labels)

fig,ax=plt.subplots(figsize=(10,6))
diabetics=(df['Diagnosis']==1).sum()
nonDiabetics=(df['Diagnosis']==0).sum()

classes=['Diabetics','Non Diabetics']
count=[diabetics,nonDiabetics]

plt.bar(classes,count,color='orange',edgecolor='black')
labeler(total,ax)
plt.title("Target Distribution")
plt.ylabel("Count")
plt.xlabel("Classes")
plt.show()

# We can see that the target is well distributed roughly 60/40

# Age: raw distribution + diabetes breakdown by age bucket, combined
def age_category(age):
    if age < 30:
        return "Under 30"
    elif age < 45:
        return "30-44"
    elif age < 60:
        return "45-59"
    else:
        return "60+"

df['Age_Category'] = df['Age'].apply(age_category)

fig, axes = plt.subplots(1, 2, figsize=(14,6))

sns.histplot(df['Age'],bins=20,shrink=0.8, edgecolor='black',color='brown', ax=axes[0])
axes[0].set_title('Age distribution')

order_age = ["Under 30","30-44","45-59","60+"]
sns.countplot(data=df, x='Age_Category', hue='Diabetes_Label', order=order_age,
              palette={'No Diabetes':'lightgray', 'Diabetes':'crimson'}, ax=axes[1])
grouped_labeler(axes[1], df, 'Age_Category', order_age)
axes[1].set_title('Diabetes Status by Age Group')

plt.tight_layout()
plt.show()
# Most of the patients of the dataset are around 30-40 or 50-60 years old

fig, axes = plt.subplots(1, 2, figsize=(14,6))

# Plot 1: Gender distribution
sns.histplot(df['Gender'], bins=2, shrink=0.8, edgecolor='black', ax=axes[0])
labeler(total, axes[0])
axes[0].set_title('Gender Distribution')

# Plot 2: Diabetes by Gender
order = ['F','M']
sns.countplot(data=df, x='Gender', hue='Diabetes_Label', order=order,
              palette={'No Diabetes':'lightgray', 'Diabetes':'crimson'}, ax=axes[1])
grouped_labeler(axes[1], df, 'Gender', order)
axes[1].set_title('Diabetes Status by Gender')
axes[1].legend(title='')

plt.tight_layout()
plt.show()
# In the dataset, women tend to have a higher percentage of diabetes 


# BMI: distribution (red/green healthy ranges, now with %) + diabetes breakdown, combined
def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Healthy'
    else:
        return 'Overweight/Obese'

df['BMI_Category'] = df['BMI'].apply(bmi_category)

fig, axes = plt.subplots(1, 2, figsize=(16,6))
bins = [10,12,14,16,18,18.5,20,22,24,24.9,26,28,30,32,34,36,38,40]  # finer near cutoffs

sns.histplot(df['BMI'], bins=bins, ax=axes[0])
labeler(total, axes[0])

for patch in axes[0].patches:
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
axes[0].legend(handles=legend_patches)
axes[0].set_title('BMI Distribution')

order = ['Underweight','Healthy','Overweight/Obese']
sns.countplot(data=df, x='BMI_Category', hue='Diabetes_Label', order=order,
              palette={'No Diabetes':'lightgray', 'Diabetes':'crimson'}, ax=axes[1])
grouped_labeler(axes[1],df,'BMI_Category',order)
axes[1].set_title('Diabetes Status by BMI Category')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()
# We can conclude that overweight/obese people are most likely to have diabetes 


# Cholesterol: raw distribution + healthy/unhealthy split (%) + diabetes breakdown, combined
def cholesterol(chol):
    if chol>5.0:
        return "Unhealthy"
    else:
        return "Healthy"
df['Chol_levels']=df['Chol'].apply(cholesterol)

fig, axes = plt.subplots(1, 3, figsize=(20,6))

sns.histplot(data=df['Chol'],bins=20,shrink=0.9,edgecolor='black', ax=axes[0])
axes[0].set_xlabel("Cholesterol levels")
axes[0].set_ylabel("Count")
axes[0].set_title("Cholesterol distribution")

sns.histplot(data=df['Chol_levels'],bins=2,color='yellow',edgecolor='black',shrink=0.9, ax=axes[1])
labeler(total,axes[1])
axes[1].set_xlabel("Cholesterol Categories")
axes[1].set_ylabel("Count")
axes[1].set_title("Cholesterol")
# We can see that a great number of patients tested have a high total cholesterol

order=['Healthy','Unhealthy']
sns.countplot(data=df,x="Chol_levels",hue='Diabetes_Label', order=order,
              palette={'No Diabetes':'lightgray', 'Diabetes':'crimson'}, ax=axes[2])
grouped_labeler(axes[2],df,'Chol_levels',order)
axes[2].set_title("Diabetes on Chol categories")

plt.tight_layout()
plt.show()


# Build class columns for TG, HDL, LDL, Creatinine and BUN before plotting
def tg_category(tg):
    if tg < 1.7:
        return "Healthy"
    else:
        return "Unhealthy"
df['TG_cat']=df['TG'].apply(tg_category)

def goodFats(hdl):
    if hdl>1:
        return "Over 1"
    else:
        return "Under 1"
df['Good_fats']=df['HDL'].apply(goodFats)

def badFats(ldl):
    if ldl<3:
        return "Under 3"
    else:
        return "Over 3"
df['Bad_fats']=df['LDL'].apply(badFats)

def creatinine(cr):
    if cr['Gender']=='F' and cr['Cr']>45 and cr['Cr']<90:
        return "Healthy"
    elif cr['Gender']=='M' and cr['Cr']>60 and cr['Cr']<115:
        return "Healthy"
    else:
        return "Unhealthy"
df['Cr_cat']=df.apply(creatinine,axis=1)

def bun(b):
    if  b['BUN']>2.5 and b['BUN']<7.1:
        return "Healthy"
    else:
        return "Unhealthy"
df['BUN_cat']=df.apply(bun,axis=1)  # was calling creatinine() before - fixed so BUN uses its own thresholds


# TG: raw distribution + healthy/unhealthy split (%) + diabetes breakdown, combined
fig, axes = plt.subplots(1, 3, figsize=(20,6))

sns.histplot(data=df['TG'],bins=20,shrink=0.9,edgecolor='black', ax=axes[0])
axes[0].set_xlabel("TG")
axes[0].set_ylabel("Count")
axes[0].set_title("Triglycerides distribution")

order_tg=["Healthy","Unhealthy"]
sns.countplot(data=df,x="TG_cat", order=order_tg, color='gold', edgecolor='black', ax=axes[1])
labeler(total, axes[1])
axes[1].set_title("TG Classes")

sns.countplot(data=df,x="TG_cat",hue='Diabetes_Label', order=order_tg,
              palette={'No Diabetes':'lightgray', 'Diabetes':'crimson'}, ax=axes[2])
grouped_labeler(axes[2],df,'TG_cat',order_tg)
axes[2].set_title("TG levels and Diabetes")

plt.tight_layout()
plt.show()


# HDL: raw distribution + healthy/unhealthy split (%) + diabetes breakdown, combined
fig, axes = plt.subplots(1, 3, figsize=(20,6))

sns.histplot(data=df['HDL'],bins=10,shrink=0.9,edgecolor='black', ax=axes[0])
axes[0].set_xlabel("HDL")
axes[0].set_ylabel("Count")
axes[0].set_title("HDL distribution")

order_hdl=["Over 1","Under 1"]
sns.countplot(data=df,x="Good_fats", order=order_hdl, color='gold', edgecolor='black', ax=axes[1])
labeler(total, axes[1])
axes[1].set_title("HDL Classes")

sns.countplot(data=df,x="Good_fats",hue='Diabetes_Label', order=order_hdl,
              palette={'No Diabetes':'lightgray', 'Diabetes':'crimson'}, ax=axes[2])
grouped_labeler(axes[2],df,'Good_fats',order_hdl)
axes[2].set_title("HDL levels and Diabetes")

plt.tight_layout()
plt.show()


# LDL: raw distribution + healthy/unhealthy split (%) + diabetes breakdown, combined
fig, axes = plt.subplots(1, 3, figsize=(20,6))

sns.histplot(data=df['LDL'],bins=10,shrink=0.9,edgecolor='black', ax=axes[0])
axes[0].set_xlabel("LDL")
axes[0].set_ylabel("Count")
axes[0].set_title("LDL distribution")

order_ldl=["Under 3","Over 3"]
sns.countplot(data=df,x="Bad_fats", order=order_ldl, color='gold', edgecolor='black', ax=axes[1])
labeler(total, axes[1])
axes[1].set_title("LDL Classes")

sns.countplot(data=df,x="Bad_fats",hue='Diabetes_Label', order=order_ldl,
              palette={'No Diabetes':'lightgray', 'Diabetes':'crimson'}, ax=axes[2])
grouped_labeler(axes[2],df,'Bad_fats',order_ldl)
axes[2].set_title("LDL levels and Diabetes")

plt.tight_layout()
plt.show()


# Creatinine: raw distribution + healthy/unhealthy split (%) + diabetes breakdown, combined
fig, axes = plt.subplots(1, 3, figsize=(20,6))

sns.histplot(data=df['Cr'],bins=10,shrink=0.9,edgecolor='black', ax=axes[0])
axes[0].set_xlabel("Cr")
axes[0].set_ylabel("Count")
axes[0].set_title("Creatinine distribution")

order_cr=["Healthy","Unhealthy"]
sns.countplot(data=df,x="Cr_cat", order=order_cr, color='gold', edgecolor='black', ax=axes[1])
labeler(total, axes[1])
axes[1].set_title("Creatinine Classes")

sns.countplot(data=df,x="Cr_cat",hue='Diabetes_Label', order=order_cr,
              palette={'No Diabetes':'lightgray', 'Diabetes':'crimson'}, ax=axes[2])
grouped_labeler(axes[2],df,'Cr_cat',order_cr)
axes[2].set_title("Creatinine levels and Diabetes")

plt.tight_layout()
plt.show()


# BUN: raw distribution + healthy/unhealthy split (%) + diabetes breakdown, combined
fig, axes = plt.subplots(1, 3, figsize=(20,6))

sns.histplot(data=df['BUN'],bins=10,shrink=0.9,edgecolor='black', ax=axes[0])
axes[0].set_xlabel("BUN")
axes[0].set_ylabel("Count")
axes[0].set_title("BUN distribution")

order_bun=["Healthy","Unhealthy"]
sns.countplot(data=df,x="BUN_cat", order=order_bun, color='gold', edgecolor='black', ax=axes[1])
labeler(total, axes[1])
axes[1].set_title("BUN Classes")

sns.countplot(data=df,x="BUN_cat",hue='Diabetes_Label', order=order_bun,
              palette={'No Diabetes':'lightgray', 'Diabetes':'crimson'}, ax=axes[2])
grouped_labeler(axes[2],df,'BUN_cat',order_bun)
axes[2].set_title("BUN levels and Diabetes")

plt.tight_layout()
plt.show()
corr = df.drop(columns="Gender").corr(numeric_only=True)

plt.figure(figsize=(8,6))
sns.heatmap(corr,
            annot=True,
            cmap="coolwarm",
            center=0)
plt.show()