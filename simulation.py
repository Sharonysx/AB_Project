# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:42:36 2020

@author: 38030
"""
import pandas as pd
import numpy as np
import os
os.chdir(r'C:\Users\38030\OneDrive\CORNELL\2020F\AB')
#%%
x1 = np.random.standard_t(3,size=50000)
x2 = np.random.standard_t(4,size=50000)
x3 = np.random.standard_t(4,size=50000)
x4 = np.random.standard_t(8,size=50000)
x5 = np.linspace(-3,3,num=50000)
np.random.shuffle(x5)

residual = np.random.standard_t(3,size=50000)
X = pd.DataFrame(data=np.array([x1,x2,x3,x4,x5,residual]).T)
X.columns = ['x1','x2','x3','x4','x5','residual']

def transform1(x):
    if x>1.2 or x<-0.5:
        return(x)
    else:
        return(0)

def transform2(x):
    if x>1.2 or x<-0.5:
        return(x**2)
    elif x<-0.8:
        return(0)
    else:
        return(2*(-x)-0.25)
        

X['x1_transform'] = X.x1.apply(transform1)
X['x2_transform'] = X.x2.apply(transform2)
X['x12_transform'] = X.x1*X.x2
X['x3_transform'] = np.sin(X['x3']/2*np.pi) #sinx function
X['x4_transform'] = X.x4.apply(lambda x: int(x>1.1))

X['y'] = X['x1_transform']*5 + X['x2_transform'] + X['x12_transform']*0.2 + X['x3_transform']*0.1 \
         + X['x4_transform']*X['x3_transform']*0.7 + -X['x4']*0.8 + X['residual']*0.25
X.to_csv('simulation_data.csv',index=False)
