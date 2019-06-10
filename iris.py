# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:56:00 2019

@author: Parth
"""

import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def sigmoid(X):
  return 1.0 / (1.0 + np.exp(-1.0 * X))
def loss(h,y,m):
    return sum(sum(-y*np.log(h)-(1-y)*np.log(1-h)))/m
def sigmoidgrad(X):
  return (1-sigmoid(X))*sigmoid(X)


data=pd.read_csv(r"D:\downloads\IRIS.csv")
data=np.array(data)

test=random.sample(range(1, 150), 120)

validation=[i for i in range(150) if not(i in test)]
y=data[:,4]
x=np.array(data[:,:4],dtype=float)
x = x-np.mean(x)
x_train=(x[test,:4])

x_train=np.concatenate((x_train,np.ones((120,1))),axis=1)


y_train=np.zeros((120,3))
for i,j in zip(test,range(120)):
    if y[i]=='Iris-setosa':
        y_train[j,0]=1
    if y[i]=='Iris-versicolor':
        y_train[j,1]=1
    if y[i]=='Iris-virginica':
        y_train[j,2]=1    

x_test=(x[validation,:4])
x_test=np.concatenate((x_test,np.ones((30,1))),axis=1)
y_test=np.zeros((30,3))

for i,j in zip(validation,range(120)):
    if y[i]=='Iris-setosa':
        y_test[j,0]=1
    if y[i]=='Iris-versicolor':
        y_test[j,1]=1
    if y[i]=='Iris-virginica':
        y_test[j,2]=1 



t=np.random.rand(5,4)
t1=np.random.rand(5,3)

h=(sigmoid(np.dot(x_train,t)))


for i in range(2000): 
    #hidden layer
    a2=(sigmoid(np.dot(x_train,t)))
    a2=np.concatenate((a2,np.ones((120,1))),axis=1)    
    
     
    #output layer
    a3=sigmoid(np.dot(a2,t1))
    #gradient
    d3=y_train-a3
    d2=(np.dot(d3,t1.T))[:,:-1]*sigmoidgrad(np.dot(x_train,t))
    
     
    D2=np.dot(a2.T,d3)/120 
    D1=np.dot(x_train.T,d2)/120
    l=loss(a3,y_train,120)    
    t1+=D2

    t+=D1
    
    plt.scatter(i,l)

a=(sigmoid(np.dot(x_train,t)))
a=sigmoid(np.dot(np.concatenate((a,np.ones((120,1))),axis=1),t1))

j=0
for i in range(120):
    if np.argmax(a[i,:])==np.argmax(y_train[i,:]):
        print("{} {}".format(a[i,:],y_train[i,:]))
        j=j+1
    
print(j/120*100)

a=(sigmoid(np.dot(x_test,t)))
a=np.concatenate((a,np.ones((30,1))),axis=1)
a=sigmoid(np.dot(a,t1))
j=0
for i in range(30):
    if np.argmax(a[i,:])==np.argmax(y_test[i,:]):
        print("{} {}".format(a[i,:],y_test[i,:]))
        j=j+1
    
print(j/30*100)






