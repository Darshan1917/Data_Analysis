# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing,svm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from matplotlib import style
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statistics import mean


x = np.array([1,2,3,4,5,6] , dtype = np.float64)
y = np.array([5,4,6,5,6,7] , dtype = np.float64)


def best_fit_slope(x,y):
    a = mean(x) * mean(y)
    b = mean(x*y)
    c = (mean(x) * mean(x))
    d = mean((x)* (x))
    z = (a-b) 
    w =(c-d)
    m = z/w
    return m


m = best_fit_slope(x,y)
print (m)
plt.scatter(x,y)
plt.show()

