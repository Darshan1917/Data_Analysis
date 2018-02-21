# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing,svm
from sklearn.feature_selection import RFE,SelectKBest,chi2,SelectPercentile
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
#%matplotlib $inline




# reading CSV as a dataframe 
Car_Data = pd.DataFrame(pd.read_csv('CarPrice_Assignment.csv', delimiter = ','))
print(Car_Data.info())
print (type(Car_Data))
print (Car_Data.tail())


# grapghing input vs price
Y_All =  np.array(Car_Data['price'])
X_All =  np.array(Car_Data['car_ID'])


print(X_All[0:5])
print(Y_All[0:5])

plt.scatter(X_All,Y_All , color='orange' , marker = '+',label = "Prices")
plt.xlabel("Car_ID")
plt.ylabel("Car_price")
plt.legend()
plt.title("Plot of Car_ID to Prices")
plt.show()




# OPerations on cleaning the data 
# 1 - Checking for null values
value = Car_Data.isnull()
print (value)
print (Car_Data.isnull().sum())

# checking for object datatype
print(Car_Data.dtypes)
print("****************")
# creating a new dataframe by dropping Car_ID , CarName and Price

Revised_car_data = Car_Data.drop(['CarName','car_ID','price'], axis=1)
print(Revised_car_data.dtypes)

# After dropping now create dummy values for categorical data

Revised_car_data = pd.get_dummies(Revised_car_data , drop_first=True)
print (Revised_car_data.head())
X_cars = np.array(Revised_car_data)
X_cars_scaled = preprocessing.scale(X_cars)
Y_cars = np.array(Car_Data['price'])



X_training  , X_testing ,Y_training , Y_testing = train_test_split(X_cars_scaled, Y_cars, test_size=0.3, random_state=1)
print ('X_train:' , X_training.shape)
print ('X_test' , X_testing.shape)
print ('Y_train' , Y_training.shape)
print ('Y_test' , Y_testing.shape)
print ("Y_test" , Y_testing)
clf = LinearRegression()
estimator = clf.fit(X_training , Y_training)
Y_pred = clf.predict(X_testing)
#print (Y_pred)
print("Y_pred" , Y_pred[0])
print ("Y_test" , Y_testing[0])
error = mean_squared_error(Y_testing, Y_pred)

#print('Coefficients: \n', clf.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(Y_testing, Y_pred))
print('Variance score: %.2f' % r2_score(Y_testing, Y_pred))
m = clf.coef_
b = clf.intercept_
Y_pred1 = m * X_testing + b
print ("formula y= {0} x + {1} :".format(m,b))

accuracy = clf.score(X_testing,Y_testing)
print (accuracy)

compared_prices = pd.DataFrame()
compared_prices['Actual_value'] = Y_testing
compared_prices['Predicted_value'] = Y_pred

print (compared_prices.head())
x= range(len(Y_testing))
print (x)
ind = np.arange(len(Y_testing))
width = 0.65

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(ind+width+0.35, Y_pred, 0.45, color='#deb0b0')

ax2 = ax.twinx()
ax2.bar(ind+width, Y_testing, width, color='#b0c4de')

ax.set_xticks(ind+width+(width/2))
ax.set_xticklabels(Y_testing)

ax.yaxis.set_ticks_position("right")
ax2.yaxis.set_ticks_position("left")

plt.tight_layout()
plt.show()

 

# New values selection
#Feature ranking with recursive feature elimination.
revised_data1 = RFE(estimator ,n_features_to_select=24,step=1)
estimator1 = revised_data1.fit(X_training , Y_training)
Y_pred_new = revised_data1.predict(X_testing)

#print (Y_pred)
print("Y_pred" , Y_pred[0])
print ("Y_test" , Y_testing[0])
print ("Y_pred_new" , Y_pred_new[0])
accuracy_new = revised_data1.score(X_testing,Y_testing)
print (accuracy_new)

# Ask someone
#X_val = revised_data1.transform(X_training)
#estimator1 = revised_data1.fit(X_val , Y_training)
#Y_pred_new1 = revised_data1.predict(X_testing)
#print("Y_pred" , Y_pred[0])
#print ("Y_test" , Y_testing[0])
#print ("Y_pred_new" , Y_pred_new1[0])
#accuracy_new = revised_data1.score(X_testing,Y_testing)
#print (accuracy_new)

#***********************************
#Univariate feature selection : means each feature is considered individually and then which has the best or highest confidence is selcted
# other methods
def f_regression(X,Y):
   import sklearn
   return sklearn.feature_selection.f_regression(X,Y,center=False)


featureSelector = SelectKBest(score_func=f_regression,k=2)
values = featureSelector.fit(X_training,Y_training)
print(values.scores_)


#Based on percentile

select = SelectPercentile(percentile=50)
select.fit(X_training,Y_training)
X_training_selected = select.transform(X_training)
X_test_selected = select.transform(X_testing)
print(X_training.shape)
print(X_training_selected.shape)

# seeing what features are selected 
mask = select.get_support()
print(mask)
plt.matshow(mask.reshape(1,-1), cmap='gray_r')

clf.fit(X_training_selected,Y_training)

Y_pred_new = clf.predict(X_test_selected)
#print (Y_pred)
print("Y_pred" , Y_pred[0])
print ("Y_test" , Y_testing[0])
print ("Y_pred_new" , Y_pred_new[0])
accuracy_new = clf.score(X_test_selected,Y_testing)
print (accuracy_new)


