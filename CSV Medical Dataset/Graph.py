import matplotlib.pyplot as plt
import numpy as np

def plot_f1score(f1_scores):
   n_groups=5
   
   fig, ax=plt.subplots()
   index=np.arange(n_groups)
   
   bar_width=0.11
   opacity=0.4
   
   g1=ax.bar(index, f1_scores['decisiontree'][:5], bar_width, alpha=opacity, color='b', label='Decision Trees')
   g2=ax.bar(index+bar_width, f1_scores['knn'][:5], bar_width, alpha=opacity, color='r', label='KNN')
   g3=ax.bar(index+(2*bar_width), f1_scores['randomforest'][:5], bar_width, alpha=opacity, color='g', label='Random Forest')
   g4=ax.bar(index+(3*bar_width), f1_scores['adaboost'][:5], bar_width, alpha=opacity, color='c', label='AdaBoost')
   g5=ax.bar(index+(4*bar_width), f1_scores['naivebayes'][:5], bar_width, alpha=opacity, color='darkred', label='Naive Bayes')
   g6=ax.bar(index+(5*bar_width), f1_scores['xgboost'][:5], bar_width, alpha=opacity, color='y', label='XGBoost')
   g7=ax.bar(index+(6*bar_width), f1_scores['mlp'][:5], bar_width, alpha=opacity, color='lime', label='MLP')
   
   ax.set_xlabel('No of runs')
   ax.set_ylabel('F1 score')
   ax.set_xticks(index+(12*bar_width)/4)
   ax.set_xticklabels(('1', '2', '3', '4', '5'))
   ax.legend()
   
   fig.tight_layout()
   plt.show()
