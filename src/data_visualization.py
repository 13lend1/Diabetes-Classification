
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

def age_category(age):
    if age < 30:
        return "Under 30"
    elif age < 45:
        return "30-44"
    elif age < 60:
        return "45-59"
    else:
        return "60+"

def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Healthy'
    else:
        return 'Overweight/Obese'

def cholesterol(chol):
    if chol>5.0:
        return "Unhealthy"
    else:
        return "Healthy"
    

def tg_category(tg):
    if tg < 1.7:
        return "Healthy"
    else:
        return "Unhealthy"
    
def goodFats(hdl):
    if hdl>1:
        return "Over 1"
    else:
        return "Under 1"
    
def badFats(ldl):
    if ldl<3:
        return "Under 3"
    else:
        return "Over 3"
    
def creatinine(cr):
    if cr['gender']==0 and cr['cr']>45 and cr['cr']<90:
        return "Healthy"
    elif cr['gender']==1 and cr['cr']>60 and cr['cr']<115:
        return "Healthy"
    else:
        return "Unhealthy"

def bun(b):
    if  b['bun']>2.5 and b['bun']<7.1:
        return "Healthy"
    else:
        return "Unhealthy"