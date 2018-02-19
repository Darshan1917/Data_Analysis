import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing,svm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from matplotlib import style
from sklearn.metrics import mean_squared_error, r2_score



# reading CSV as a dataframe 
data = pd.DataFrame(pd.read_csv('CarPrice_Assignment.csv', delimiter = ','))
#print (type(data))
#print (data.tail())       
value = data.isnull()
print (data.isnull().sum())

# what does inplace mean
#print(data.dtypes)
data = data.drop(['CarName','car_ID'], axis=1)
obj_df = data.select_dtypes(include=['object']).copy()
obj_df1 = data.select_dtypes(exclude=['object']).copy()
print (obj_df1.head())

#  
#fueltype_mapping = {'gas':1,'diesel':0}
#data['fueltype'] =  data['fueltype'].map(fueltype_mapping)
#
## we can have inverse mapping 
#inv_map = {v:k for k , v in fueltype_mapping.items()} 
#data['fueltype'] =  data['fueltype'].map(inv_map)
#
# 
##print(np.unique(obj_df['aspiration']))
#class_mapping = {label:idx for idx , label in enumerate(np.unique(obj_df['aspiration']))}
##print(class_mapping)
#
#print(np.unique(obj_df['enginetype']))
#
##type one 
data_dummies = pd.get_dummies(obj_df)
print (data_dummies.head())
features = data_dummies

# methond 2
#label_encoder = LabelEncoder()
#integer_encoded = label_encoder.fit_transform(obj_df['drivewheel'])
#print(integer_encoded)
#
#
#onehot_encoder = OneHotEncoder(sparse=False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#print(onehot_encoded)


#### NOT RELEASED THERE IN GIT DOWNLOAD
# LINK :pip install git+git://github.com/scikit-learn/scikit-learn.git
#onehot_encoder = CategoricalEncoder(encoding='onehot', categories='auto')
#integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#print(onehot_encoded)
Frame = [obj_df1,features]
X = pd.concat(Frame)
print (X.head())

















 