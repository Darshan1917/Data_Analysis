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



# reading CSV as a dataframe 
data = pd.DataFrame(pd.read_csv('CarPrice_Assignment.csv', delimiter = ','))
#print (type(data))
#print (data.tail())       
value = data.isnull()
print (data.isnull().sum())

# what does inplace mean
print(data.dtypes)
data = data.drop(['CarName','car_ID'], axis=1)
#data = data.select_dtypes(include=['object']).copy()
obj_df1 = data.select_dtypes(include=['object']).copy()
#print (np.unique(obj_df1['aspiration']))
#print (pd.get_dummies(obj_df1['aspiration']))
# when we use get dummies the values gets over written if we assign it to same value
print ("before:" ,obj_df1.head())
print("####################")
obj_df1 = pd.get_dummies(obj_df1)
#print (obj_df1['doornumber']) # uncomment and check error
print ("After", obj_df1.head())
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
data_dummies = pd.get_dummies(data , drop_first=True)
print (data_dummies.head())
X = np.array(data_dummies.drop(['price'],1))
X = preprocessing.scale(X)
Y = np.array(data_dummies['price'])


X_train , X_test ,Y_train , Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
print ('X_train:' , X_train.shape)
print ('X_test' , X_test.shape)
print ('Y_train' , Y_train.shape)
print ('Y_test' , Y_test.shape)
clf = LinearRegression()
clf.fit(X_train , Y_train)
Y_pred = clf.predict(X_test)
#print (Y_pred)
print("Y_pred" , Y_pred[0])
print ("Y_test" , Y_test[0])
#print('Coefficients: \n', clf.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_pred))
print('Variance score: %.2f' % r2_score(Y_test, Y_pred))
m = clf.coef_[0]
b = clf.intercept_
print ("formula y= {0} x + {1} :".format(m,b))

#plt.scatter(X_test, Y_test,  color='black')
#plt.plot(X_test, Y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
accuracy = clf.score(X_test,Y_test)
print (accuracy)



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

















 