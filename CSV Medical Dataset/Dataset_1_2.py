import pandas as pd
import DataPreprocessing as dp
import F1_Score as fs
import Test_Percentage as tp

# dataset 1 - diabetic dataset
df = pd.read_csv("diabetic_data.csv")
df = df.sample(1000) 
df = df.fillna(0)
   
df.loc[df.loc[:,'readmitted']!='<30','readmitted']=0
df.loc[df.loc[:,'readmitted']=='<30','readmitted']=1
   
df.drop(['weight', 'medical_specialty','payer_code'],axis=1,inplace=True)
df.drop(['encounter_id','patient_nbr'],axis=1,inplace=True)

df = dp.data_preprocess(df)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

fs.find_f1(X, y)
tp.find_test(X, y)

# dataset 2 - rhc dataset
df = pd.read_csv("rhc.csv")
df = df.sample(1000)
df = df.fillna(0)

df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.loc[df.loc[:,'death']=='No','death']=0
df.loc[df.loc[:,'death']=='Yes','death']=1

df = dp.data_preprocess(df)

col = df.columns
ind = col.drop(['death'])

X = df.loc[:, col]
y = df.loc[:, 'death']

fs.find_f1(X, y)  
tp.find_test(X, y) 

