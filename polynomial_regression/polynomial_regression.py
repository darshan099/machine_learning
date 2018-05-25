#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:27:22 2018


"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#getting csv file
dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#since we do not have a lot of data here so we dont differentiate b/w train
#and test set

#fitting linear regresssion to dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)

#create another linear regression
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

#visualizing polynomial regression results
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color='blue')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#predicting from linear regression
lin_reg.predict(6.5)

#predicting from polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))