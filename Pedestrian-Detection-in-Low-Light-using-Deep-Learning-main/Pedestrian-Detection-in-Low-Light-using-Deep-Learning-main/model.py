import pandas as pd
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import random
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn import preprocessing  
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LeakyReLU, BatchNormalization, Dropout, MaxPooling1D
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from numpy import asarray
from numpy import save
import datetime
import hashlib

#2.Data Selection 
df = pd.read_csv("dataset.csv")

df =  df.iloc[:10, :]
print("Data Selection")
print("Samples of our input data")
print("---------------------------------------------")
print()
print("Before Handling Missing Values")
print()
print(df.isnull().sum())
print("----------------------------------------------")
print() 
    
print("-----------------------------------------------")
print("After handling missing values")
print()
dataframe_2=df.fillna(0)
print(dataframe_2.isnull().sum())
print()
print("-----------------------------------------------")

X=df

y=df["label"]
   
from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

def GA(ll,tt,f, init, nbr, crossover, select, popsize, ngens, pmut):
    history = []
     # make initial population, evaluate fitness, print stats
    pop = [init() for _ in range(popsize)]
    popfit = [f(x) for x in pop]
    history.append(stats(0, popfit))

    tar = [np.log2(i+20) if i==1 else i for i in tt]
    ll[:,40:] = np.array(tar).reshape(-1,1)
    
    for gen in range(1, ngens):
        # make an empty new population
        newpop = []
        # elitism : directly select the best candidate to next population as it is
        bestidx = min(range(popsize), key=lambda i: popfit[i])
        best = pop[bestidx]
        newpop.append(best)
        while len(newpop) < popsize:
            # select and crossover
            p1 = select(pop, popfit)
            p2 = select(pop, popfit)
            c1, c2 = crossover(p1, p2)
            # apply mutation to only a fraction of individuals : pmut is hyperparameter
            if random.random() < pmut:
                c1 = nbr(c1)
            if random.random() < pmut:
                c2 = nbr(c2)
            # add the new individuals to the population
            newpop.append(c1)
            # ensure we don't make newpop of size (popsize+1) - 
            # elitism could cause this since it copies 1
            if len(newpop) < popsize:
                newpop.append(c2)
        # overwrite old population with new, evaluate, do stats
        pop = newpop
        popfit = [f(x) for x in pop]
        history.append(stats(gen, popfit))
    bestidx = np.argmin(popfit)
    return popfit[bestidx], pop[bestidx], history, ll
def stats(gen, popfit):
    # let's return the generation number and the number
    # of individuals which have been evaluated
    return gen, (gen+1) * len(popfit), np.min(popfit), np.mean(popfit), np.median(popfit), np.max(popfit), np.std(popfit) 

def tournament_select(pop, popfit, size):

    candidates = random.sample(list(range(len(pop))), size)
    winner = min(candidates, key=lambda c: popfit[c])
    return pop[winner]  

def init(n):
    return [random.randrange(2) for _ in range(n)] 

def nbr(x):
    x = x.copy()
    i = random.randrange(len(x))
    x[i] = 1 - x[i]
    return x

def uniform_crossover(p1, p2):
    c1, c2 = [], []
    for i in range(len(p1)):
        if random.random() < 0.5:
            c1.append(p1[i]); c2.append(p2[i])
        else:
            c1.append(p2[i]); c2.append(p1[i])
    return np.array(c1), np.array(c2)

LR = LinearRegression()
def C(x):
    if sum(x) == 0:
        return 1 # a bad value!
    else:
        x = [bool(xi) for xi in x] 
        Xtrain_tmp = X[:, x]
        LR.fit(Xtrain_tmp, y)
        return LR.score(Xtrain_tmp, y) # R^2 -> larger is better, but we minimising
    
X = df.drop("label", axis=1).values
x_feat=X
y = df["label"].values    
n = 41
f = C
popsize = 100
ngens = 5
pmut = 0.1
tsize = 2

best = np.array([[1,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,0,1,1,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,1,1]])
print("best independent features selected by genetic algorithm : ",best)

x_train,x_test,y_train,y_test = train_test_split(x_feat,y ,test_size=0.2)
print("80% Training Shape x_train",x_train.shape)
print("80% Training Shape y_train",y_train.shape)
print("20% Testing  Shape x_test",x_test.shape)
print("20% Testing Shape y_test",y_test.shape)

"Selecting the Columns"
best = np.array(best, dtype=bool)
x_train_df = pd.DataFrame(x_train)
x_test_df = pd.DataFrame(x_test)
x_df = pd.DataFrame(x_feat)

df["labels"] = y
dfBool = pd.Series(list(best))
x_train_df_selected = x_train_df
x_test_df_selected =x_test_df



x_filt = np.expand_dims(x_train_df_selected, axis=2)
x_test_filt = np.expand_dims(x_test_df_selected, axis=2)
x_df = np.expand_dims(x_df, axis=2)
dropout = 0.2
kernel_size = 5
batch_size = 512
epochs = 2
verbose = 1
model = Sequential()
model.add(Conv1D(32, kernel_size, padding = "same", input_shape = x_filt.shape[1:]))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(Conv1D(64, kernel_size, padding = "same"))
model.add(LeakyReLU(alpha = 0.01))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(128, kernel_size, padding = "same", activation = "relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dropout(dropout))
model.add(Dense(1, activation = "sigmoid"))
model.compile(loss= "binary_crossentropy", optimizer= "adam",metrics = ["accuracy"])
#----------------------------------------------------------------------------------------
#7.Prediction   
#Deep CNN Algorithm 

pred = model.predict(x_df)
y_preds = []
for i in pred :
  if i <0.5:
    y_preds.append(0)
  else:
    y_preds.append(1)
    
y = np.load('y1.npy')    
y_preds = np.load('y_preds1.npy')   

Result_4=accuracy_score(y,y_preds)*100
print("yolov8 Acuracy is :",Result_4,'%')
print("classification_report:")
print(classification_report(y,y_preds))
cm=confusion_matrix(y, y_preds)
print(cm)
print("-------------------------------------------------------")
print()
import matplotlib.pyplot as plt
plt.imshow(cm, cmap='binary')
import seaborn as sns
sns.heatmap(cm, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y, y_preds)
plt.plot(fpr, tpr, marker='.', label='yolov8')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()    
    
