import re
import nltk

fp1=open(r"2010\concept_assertion_relation_training_data.tar\concept_assertion_relation_training_data\beth\txt\record-179.txt", 
        "r")

fp2=open(r"2010\concept_assertion_relation_training_data.tar\concept_assertion_relation_training_data\beth\concept\record-179.con", 
         "r")

op=open("output.csv", "a")

lines1=fp1.readlines()
lines2=fp2.readlines()

ind=lines1.index('Service :\n') # find the index of "Service :\n" and get all data after that
lines1=lines1[ind+1: ]

count=0
tee=""

# function to get line no, start and end index of medical term
def split_line(l):
   l = l.split("||") 
    x = l[0] 
    i = re.search("\d+:\d+ \d+:\d+", x) 
    i = int(i.start())
    l = x[i:] 
    
    a = l.split(" ")
    
    n, s=a[0].split(":")
    n, e=a[1].split(":")
    n=int(n)
    s=int(s)
    e=int(e)
    
    return n, s, e

# function to get tag of medical term
def find(l):
    global tee
   
    l = l.split("||")
    
    label = re.findall(r'"(.*?)"', l[1])
    label = ''.join(label) 
    
    if label=="problem":
        tee = "P"
    elif label=="test":
        tee = "T"
    elif label=="treatment":
        tee = "M"
        
    return tee
    
for l in lines2:
    count = count+1
    
    n, s, e = split_line(l) 
    words = lines1[(n-1-(ind+1))].split()
    
    tag = ['O']*len(words)
    
    for i in range(s, e+1):
        tags = find(l)
        tag[i] = tags
        
    tagged = nltk.pos_tag(words) 
    
    for (w, ti, ta) in zip(words, tag, tagged): 
        line = "Sentence: "+str(count)+','+'"'+str(w)+'"'+','+'"'+str(ta[1])+'"'+','+str(ti)+'\n'
        op.write(line) # writing to output file
    
fp1.close()
fp2.close()
op.close()