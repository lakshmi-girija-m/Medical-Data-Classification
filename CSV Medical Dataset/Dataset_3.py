import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential

import DataPreprocessing as dp
import F1_Score as fs
import Test_Percentage as tp
import Graph as gh
import GeneticAlgorithms as ga

# dataset 3 - rhc dataset
df = pd.read_csv("vlbw.csv")
df=df.fillna(0)

df = dp.data_preprocess(df)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

f1_scores = {}
test_f1_scores = {}
algorithms = ['adaboost', 'decisiontree', 'knn', 'naivebayes', 'randomforest', 'xgboost', 'mlp']
test_range = np.round(np.arange(0.1, 1, 0.1), 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   
# F1 score for 10 runs
for algorithm in algorithms:
   model = fs.create_model(algorithm)
   f1score=[]
   print("{}: ".format(algorithm))
      
   for r in range(10):
      if isinstance(model, Sequential):
       model = ga.get_model(X_train, y_train)
       
       y_classes = model.predict_classes(X_test, verbose=0)
       y_pred = y_classes[:, 0]
    
      else:
         model.fit(X_train, y_train)
         y_pred = model.predict(X_test)
      
      f1 = metrics.f1_score(y_test, y_pred)
      print("Run -> {0}   ---   F1 score -> {1}".format(r+1, f1))
      f1score.append(f1)
         
   f1_scores[algorithm] = f1score
      
gh.plot_f1score(f1_scores)
   
# F1 score for different test sizes
for algorithm in algorithms:
   model = tp.create_model(algorithm)
   f1score=[]
   print("{}: ".format(algorithm))
      
   for t in test_range:
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t)
         
      if isinstance(model, Sequential):
         model = ga.get_model(X_train, y_train)
       
         y_classes = model.predict_classes(X_test, verbose=0)
         y_pred = y_classes[:, 0]
    
      else:
         model.fit(X_train, y_train)
         y_pred = model.predict(X_test)
      
      f1 = metrics.f1_score(y_test, y_pred)
      print("Run -> {0}   ---   F1 score -> {1}".format(r+1, f1))
      f1score.append(f1)
         
   test_f1_scores[algorithm] = f1score
      
gh.plot_f1score(test_f1_scores)
   