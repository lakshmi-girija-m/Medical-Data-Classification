import GeneticAlgorithms as ga
import DataPreprocessing as dp
import Graph as gh

import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def create_model(algorithm):
   if algorithm=='adaboost':
      model = AdaBoostClassifier(n_estimators=50, learning_rate=1)
   elif algorithm=='decisiontree':
      model = DecisionTreeClassifier(max_depth=2) 
   elif algorithm=='knn':
      model = KNeighborsClassifier(n_neighbors=2)
   elif algorithm=='naivebayes':
      model = GaussianNB()
   elif algorithm=='randomforest':
      model = RandomForestClassifier(n_estimators=50, max_features=0.4) 
   elif algorithm=='xgboost':
      model = XGBClassifier(booster='gbtree', max_depth=5)
   elif algorithm=='mlp':
      model = Sequential()
      
   return model

def find_test(X, y):
   f1_scores = {}
   algorithms = ['adaboost', 'decisiontree', 'knn', 'naivebayes', 'randomforest', 'xgboost', 'mlp']
   test_range = np.round(np.arange(0.1, 1, 0.1), 1)
   
   # F1 score for different test sizes
   for algorithm in algorithms:
      model = create_model(algorithm)
      f1score=[]
      print("{}: ".format(algorithm))
      
      for t in test_range:
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t)
         
         f1 = ga.genetic_function(model, X_train, X_test, y_train, y_test)
         print("Test size -> {0}   ---   F1 score -> {1}".format(t, f1))
         f1score.append(f1)
         
      f1_scores[algorithm] = f1score
      
   gh.plot_f1score(f1_scores)
   
