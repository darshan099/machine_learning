#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 08:14:16 2018

@author: darshan
"""

import matplotlib.pyplot as plt
import pandas as pd
#import dataset
dataset=pd.read_csv('salary.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#train the values
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#simple linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting the result
y_pred=regressor.predict(x_test)

#plot actual value vs predicted values
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.xlabel('salary')
plt.ylabel('experience')
plt.title('actual vs predicted')
plt.show()

