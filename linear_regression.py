# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:41:04 2018

@author: dumapath
"""

# Based on dataset given by boston_load in sklearn  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import scipy.stats as stats

# the data will we avialable of sklearn python module 
from sklearn.datasets import load_boston
boston = load_boston()
# class of util bunch
print (boston.keys())

print (boston.feature_names)

print (boston.target)
print (boston.DESCR) # describes the dataset ehat is the functionality

bos  = pd.DataFrame(boston.data)
print (bos.head())
print (bos.tail())

