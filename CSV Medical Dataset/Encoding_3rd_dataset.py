import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import DataPreprocessing as dp
import GeneticAlgorithms as ga

df = pd.read_csv("vlbw.csv")
df.drop(['lol'],axis=1,inplace=True)
df = df.fillna(0)

# One hot encoding
data1 = dp.data_preprocess(df)

# Label Encoding
label_ = preprocessing.LabelEncoder()
df = df.astype(str)
data2 = df.apply(label_.fit_transform)

model = RandomForestClassifier(n_estimators=50, max_features=0.4) 
f1_scores = {}
index = 0

for df in (data1, data2):    
   f1 = []
   for i in range(10):
       X = df.iloc[:, :-1]
       y = df.iloc[:, -1]
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
       
       score = ga.genetic_function(model, X_train, X_test, y_train, y_test)
       f1.append(score)
   f1_scores[index] = f1
   index+=1
   
import matplotlib.pyplot as plt
plt.axis([1, 10, 0.5, 1.1])
plt.xlabel('No of runs')
plt.ylabel('F1 score')
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], f1_scores[0],'r-', label='Onehot Encoding')
plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], f1_scores[1],'b-', label='Label Encoding')
plt.legend()
plt.show()
