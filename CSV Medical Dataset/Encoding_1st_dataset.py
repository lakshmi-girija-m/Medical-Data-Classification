import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import DataPreprocessing as dp
import GeneticAlgorithms as ga

df = pd.read_csv("diabetic_data.csv")
df = df.sample(1000)
df=df.fillna(0)

df.loc[df.loc[:,'readmitted']!='<30','readmitted']=0
df.loc[df.loc[:,'readmitted']=='<30','readmitted']=1

df.drop(['weight', 'medical_specialty','payer_code'],axis=1,inplace=True)
df.drop(['encounter_id','patient_nbr'],axis=1,inplace=True)

# One hot encoding
data1 = dp.data_preprocess(df)

# Label Encoding
label_ = preprocessing.LabelEncoder()
data2 = df.apply(label_.fit_transform)

# Mannual Encoding
df.loc[df.age== '[0-10)','age'] = 0
df.loc[df.age== '[10-20)','age'] = 10
df.loc[df.age== '[20-30)','age'] = 20
df.loc[df.age== '[30-40)','age'] = 30
df.loc[df.age== '[40-50)','age'] = 40
df.loc[df.age== '[50-60)','age'] = 50
df.loc[df.age== '[60-70)','age'] = 60
df.loc[df.age== '[70-80)','age'] = 70
df.loc[df.age== '[80-90)','age'] = 80
df.loc[df.age== '[90-100)','age'] = 90
df.age = df.age.astype(np.int32)

df.loc[df.max_glu_serum== 'None','max_glu_serum'] = 0
df.loc[df.max_glu_serum== 'Norm','max_glu_serum'] = 100
df.loc[df.max_glu_serum== '>200','max_glu_serum'] = 200
df.loc[df.max_glu_serum== '>300','max_glu_serum'] = 300
df.max_glu_serum = df.max_glu_serum.astype(np.int32)

df.loc[df.A1Cresult== 'None','A1Cresult'] = 0
df.loc[df.A1Cresult== 'Norm','A1Cresult'] = 5
df.loc[df.A1Cresult== '>7','A1Cresult'] = 7
df.loc[df.A1Cresult== '>8','A1Cresult'] = 8
df.A1Cresult = df.A1Cresult.astype(np.int32)

df.loc[df.change== 'No','change'] = 0
df.loc[df.change== 'Ch','change'] = 1
df.change = df.change.astype(np.int8)

df.loc[df.diabetesMed== 'No','diabetesMed'] = 0
df.loc[df.diabetesMed== 'Yes','diabetesMed'] = 1
df.diabetesMed = df.diabetesMed.astype(np.int8)
        
medications = ["metformin", "repaglinide", "nateglinide", 
           "chlorpropamide", "glimepiride", "acetohexamide", 
           "glipizide", "glyburide", "tolbutamide", "pioglitazone", 
           "rosiglitazone", "acarbose", "miglitol", "troglitazone", 
           "tolazamide", "examide", "citoglipton", "insulin", 
           "glyburide-metformin", "glipizide-metformin", 
           "glimepiride-pioglitazone", "metformin-rosiglitazone", 
           "metformin-pioglitazone"]
for med in medications:
    df.loc[df[med] == 'No', med] = -20
    df.loc[df[med] == 'Down', med] = -10
    df.loc[df[med] == 'Steady', med] = 0
    df.loc[df[med] == 'Up', med] = 10
    df[med] = df[med].astype(np.int32)
    
categoricals = ['race', 'gender', 'diag_1', 'diag_2', 'diag_3']
for c in categoricals:
    df[c] = pd.Categorical(df[c]).codes
    
model = RandomForestClassifier(n_estimators=50, max_features=0.4) 
f1_scores = {}
index = 0

for data in (data1, data2, df):    
   f1 = []
   for i in range(10):
       X = data.iloc[:, :-1]
       y = data.iloc[:, -1]
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
       score = ga.genetic_function(model, X_train, X_test, y_train, y_test)
       f1.append(score)
   f1_scores[index] = f1
   index+=1
    
import matplotlib.pyplot as plt
plt.axis([1, 10, 0, 1.0])
plt.xlabel('No of runs')
plt.ylabel('F1 score')
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], f1_scores[0],'r-', label='Onehot Encoding')
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], f1_scores[1],'b-', label='Label Encoding')
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], f1_scores[2],'g-', label='Manual Encoding')
plt.legend()
plt.show()