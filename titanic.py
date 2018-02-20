# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np


titanic = pd.read_csv('train.csv', delimiter = ',')
'''
## checking the datatypes 
#print (titanic.info())
## or 
#print (titanic.dtypes)
## Describe gives mean total number , median etc  
#print (titanic.describe())
'''
#    print (type(titanic))

''' to see head and tail of a dataset'''
print(titanic.head()) # gives first 5 record 
print(titanic.tail()) # gives last 5 records of the dataset


''' shape of my dataset '''
print(titanic.shape)
value = titanic.isnull()
print (value)
print (titanic.isnull().sum())

''' General way to create a dataframe  pd.Dataframe()
'''

df = pd.DataFrame()
df['Name'] = ['Steve Smith','Virat Kholi', 'Sachin ramesh Tendulkar']
df['age'] =  [30,29,37]
df['salary'] = [35000,45000,55000]
df['country'] = ['Aus','Ind','Ind']

print (df)

#adding rows to dataframe
#create a row and add
new_row = pd.Series([' Angelo Mathews', 28, 32000,'Sri'], index =['Name','age','salary','country'])
#print(new_row)
#print(type(new_row))

#we use df.append
df = df.append(new_row, ignore_index=True)
print (df)  


#navigate in dataframe loc and iloc

print(df.iloc[0])
print(titanic.iloc[0][4])

''' select the entire row '''
print(df.iloc[ : 2]) # only select only 2 rows
print(df[['age','salary']]) #only columns
print(df.loc[:,"age":"country"])
print(df.iloc[:,1:3])




