# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:42:36 2020

@author: 38030
"""
import pandas as pd
import numpy as np
import os
import gc
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
p = matplotlib.rcParams
p["font.size"] = 20
p["axes.unicode_minus"] = False
p['lines.linewidth'] = 3
p['pdf.fonttype'] = 42
p['ps.fonttype'] = 42
p["figure.figsize"] = [12, 8]
p['grid.color'] = 'k'
p['grid.linestyle'] = ':'
p['grid.linewidth'] = 0.5
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, 5))  
import seaborn as sns
sns.set(style="whitegrid")

os.chdir(r'C:\Users\38030\OneDrive\CORNELL\2020F\AB')
#%%
x1 = np.random.normal(0,size=50000)
x2 = np.random.normal(0,size=50000)
x3 = np.random.normal(0,size=50000)
x4 = np.random.standard_t(8,size=50000)

x5 = np.linspace(-3,3,num=50000)
np.random.shuffle(x5)

residual = np.random.standard_t(10,size=50000)
X = pd.DataFrame(data=np.array([x1,x2,x3,x4,x5,residual]).T)
X.columns = ['x1','x2','x3','x4','x5','residual']

def transform1(x):
    if x>1.2 or x<-0.5:
        return(x)
    else:
        return(0)

def transform2(x):
    if 5>x>1.2 or x<-1:
        return(x**2/2)
    elif x>-2.5:
        return(x/2-1)
    else:
        return(0)
        
def transform45(x):
    return(int(x.x4>1 and x.x5<-0.7))       

X['x1_transform'] = X.x1.apply(transform1)
X['x2_transform'] = X.x2.apply(transform2)
X['x45_transform'] = X.apply(transform45,axis=1)
X['x3_transform'] = 0.5*(X.x3-0.5)**2 #sinx function
X['x4_transform'] = np.sin(X.x4)

X['y'] = X['x1_transform']*5 + X['x3_transform'] - X['x45_transform']*3\
         + X['x2_transform']*2 - X['x4_transform'] + X['residual']
         
X.to_csv('simulation_data.csv',index=False)

plt.scatter(X['x1'],X['y'])
plt.scatter(X['x1'],X['x1_transform']*5)
plt.show()



