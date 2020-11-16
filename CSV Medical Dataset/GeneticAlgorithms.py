import statistics
import pandas as pd
from sklearn import metrics
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def get_model(X_train, y_train):
    (nr, nc)=X_train.shape
    model = Sequential()
    model.add(Dense(20, input_dim=nc, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, verbose=0)
    return model


def index_select(r, ft):
    for a in range(len(ft)): 
        if r>=ft[a] and r<=ft[a+1]:
            return a+1
 
   
def selection(n, popu, fit):
    indices=[]
    pops = []
    fitness = []
    ft=np.asarray(fit).cumsum()
    
    for i in range(n):
        r = random.uniform(0, ft[-1]) 
        indices.append(index_select(r, ft))

    for i in indices:
       if i==None:
          i=0
       pops.append(popu[i-1])
       fitness.append(fit[i-1])
    
    return pops, fitness


def generate_population(nc):
    items=[0, 1]
    # generate a chromosome with 0s and 1s which tell whether to select a feature or not
    chromosome = np.random.choice(items, size=nc, replace=True)
    return chromosome


# fitness function, which returns the F1 score for each chromosome
def find_fittness(model, chromosome, X_train, y_train, X_test, y_test):
    X = pd.DataFrame(data=X_train.iloc[:,np.nonzero(chromosome)[0]]) 
    
    if isinstance(model, Sequential):
       model = get_model(X, y_train)
       y_classes = model.predict_classes(X_test.iloc[:,np.nonzero(chromosome)[0]], verbose=0)
       y_pred = y_classes[:, 0]
    
    else:
       model.fit(X, y_train)
       y_pred = model.predict(X_test.iloc[:,np.nonzero(chromosome)[0]])
       
    f_score = metrics.f1_score(y_test, y_pred)
    
    return f_score


def cross_over(ch1, ch2):
    size = min(len(ch1), len(ch2))
    point = random.randint(0, size-1)
    ch1[point:], ch2[point:] = ch2[point:], ch1[point:]
    
    return ch1, ch2
    
 
def mutate(ch):
    ran=random.randint(0, len(ch)-1)
    
    for i in range(ran):
        p = random.randint(0, len(ch)-1)
        if ch[p]==1:
            ch[p]=0
        else:
            ch[p]=1
            
    return ch


def genetic_function(model, X_train, X_test, y_train, y_test):  
    population = []
    fittness=[0]
    child_f=[]
    children=[]
    l=0
    averages=[]
    
    # generating chomosomes to create an initial population of size 10
    for i in range(10):
        population.append(generate_population(X_train.shape[1]))

    # for each chromosome find the fittest one based on the F1 score
    for chromosome in population:
        f1_score=find_fittness(model, chromosome, X_train, y_train, X_test, y_test)
        fittness.append(f1_score)

    while l<5:
       
        children.clear() 
        child_f.clear() 
        
        # selecting 2 parent chromosomes to produce next generation using cross over
        par, fit = selection(2, population, fittness)

        for i in range(5):
            child1, child2 = cross_over(par[0], par[1])
            children.append(child1)
            children.append(child2)
            
        for i in range(5):
            par, fit = selection(1, population, fittness)
            child3 = mutate(par[0])
            ch=np.squeeze(child3)
            children.append(ch)
         
        # for each child chromosome created find it's fitness score, i.e, F1 score
        for child in children:
            
            f1_score=find_fittness(model, child, X_train, y_train, X_test, y_test)
            child_f.append(f1_score)

        if max(fittness) > max(child_f):
           
            m_ind=np.argmax(fittness)
            fittest=population[m_ind-1]
            m_fit =fittness[m_ind-1] 

        else:
            m_ind=np.argmax(child_f)
            fittest=children[m_ind-1]
            m_fit =child_f[m_ind-1] 
    
        population, fittness=selection(len(population), population+children, fittness+child_f)
        
        population[0]=fittest
        fittness[0] = m_fit

        averages.extend(fittness)
        l=l+1
        
    f1=statistics.mean(averages) 
    
    return f1