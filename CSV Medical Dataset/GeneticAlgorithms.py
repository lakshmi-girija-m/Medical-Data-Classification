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

'''
r value is randomly generated. We find the elements between which r lies and get the index of the upper range element. We are doing this to select a chromosome to produce next generation.
'''
def index_select(r, ft):
    for a in range(len(ft)): 
        if r>=ft[a] and r<=ft[a+1]:
            return a+1
    
'''
The uniform() method returns a random floating number between the two specified numbers (both included).

The asarray() menthod is used to convert the input to an array.

The cumsum() returns the cumulative sum of the elements along a given axis. Default value for axis is None. In this case cummulative sum for flattened array is returned.
'''
     
def selection(n, popu, fit):
    indices=[]
    pops = []
    fitness = []
    
    # finding cummulative sum for fitnes score array
    ft=np.asarray(fit).cumsum()
    
    for i in range(n):
        r = random.uniform(0, ft[-1]) # return a value between 0 and last element of cummulative sum array
        indices.append(index_select(r, ft))

    for i in indices:
       if i==None:
          i=0
          
    # return the chromosomes from population corresponding to the indices
       pops.append(popu[i-1])
       fitness.append(fit[i-1])
    
    return pops, fitness

# generate chromosomes
def generate_population(nc):
    items=[0, 1]
    # generate a chromosome with 0s and 1s which tell whether to select a feature or not
    chromosome = np.random.choice(items, size=nc, replace=True)
    return chromosome

# fitness function, which returns the F1 score for each chromosome
def find_fittness(model, chromosome, X_train, y_train, X_test, y_test):
   
    # based on the chromosome, select the features for which gene is 1
    X = pd.DataFrame(data=X_train.iloc[:,np.nonzero(chromosome)[0]]) 
    
    if isinstance(model, Sequential):
       model = get_model(X, y_train)
       
       y_classes = model.predict_classes(X_test.iloc[:,np.nonzero(chromosome)[0]], verbose=0)
       y_pred = y_classes[:, 0]
    
    else:
       model.fit(X, y_train)
       y_pred = model.predict(X_test.iloc[:,np.nonzero(chromosome)[0]])
       
    f_score = metrics.f1_score(y_test, y_pred)
    
    return f_score # F1 score or fitness score for a chromosome


def cross_over(ch1, ch2):
   
    size = min(len(ch1), len(ch2))
    point = random.randint(0, size-1) # select a random point in the chromosome
    
    # second half of chromosome 2 is swapped with second half of chromosome 1
    # second half of chromosome 1 is swapped with second half of chromosome 2
    ch1[point:], ch2[point:] = ch2[point:], ch1[point:]
    return ch1, ch2
    
 
def mutate(ch):
    # changing random bits in a chromosome at index p, ran no of times
    
    # no of times we want to change bits
    ran=random.randint(0, len(ch)-1)
    
    for i in range(ran):
        # generating a random index where we want to change bit
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
        
        # fitness scores for the respective chromosomes
        fittness.append(f1_score)
    
   # stopping condition
    while l<5:
       
        children.clear() # child chromosomes
        child_f.clear() # F1 scores for child chromosome
        
        # selecting 2 parent chromosomes to produce next generation using cross over
        par, fit = selection(2, population, fittness)

        for i in range(5):
            child1, child2 = cross_over(par[0], par[1])
            children.append(child1)
            children.append(child2)
            
        for i in range(5):
           
            # selecting 1 parent chromosome to produce next generation using mutation
            par, fit = selection(1, population, fittness)
            child3 = mutate(par[0])
            ch=np.squeeze(child3)
            children.append(ch)
         
        # for each child chromosome created find it's fitness score, i.e, F1 score
        for child in children:
            
            f1_score=find_fittness(model, child, X_train, y_train, X_test, y_test)
            child_f.append(f1_score)
        
        # fitness - F1 scores of initial population
        # child_f - F1 scores of children
        
        # if initial population has best fit score
        if max(fittness) > max(child_f):
           
            m_ind=np.argmax(fittness) # get index of the score
            fittest=population[m_ind-1] # get chromosome corresponding to that index
            m_fit =fittness[m_ind-1] # get fitness score corresponding to that index
         
        # if children has best fit score
        else:
            # get best fit value and the respective child chromosome
            m_ind=np.argmax(child_f)
            fittest=children[m_ind-1]
            m_fit =child_f[m_ind-1] 
    
        # out of initial population and children select 10 chromosomes for next generation
        population, fittness=selection(len(population), population+children, fittness+child_f)
        
        # add the fittest chromosome and fitness score to this next population
        population[0]=fittest
        fittness[0] = m_fit
        
        # store all fitness values for each generation
        averages.extend(fittness)
        l=l+1
        
    f1=statistics.mean(averages) # finding average of all generation's fitness scores
    
    return f1