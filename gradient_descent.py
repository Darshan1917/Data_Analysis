# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_error(b,m,points):
    total_error = 0
    for i in range (0, len(points)):
        # get x value
        x = points[i,0]
        y = points[i,1]
        total_error += (y-((m*x + b)**2))
    return total_error/float(len(points))

def Gradient_descent_runner(points,b,m,learning_rate,num_iteration):
    b = b
    m = m
    for i in range(num_iteration):
        b , m = step_gradient (b,m,np.array(points),learning_rate)
    return[b,m]
    
def step_gradient(b_current,m_current,points,learning_rate):
    m_gradient = 0
    b_gradient = 0
    n = float(len(points))
    for i in range(0 , len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += - (2/n) * (y - (( m_current * x) + b_current)) 
        m_gradient += (2/n) * x * (y - (( m_current * x) + b_current))
        
    # update b,m using partial derivative
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return new_b,new_m
        

# Read the data
points  = pd.read_csv('train1.csv' , delimiter =',')
# Learning rate:
learning_rate = 0.001
initial_b = 0
initial_m = 0
num_iteration = 1000
X = points[0]
y = points[1]
plt.plot(X,y)
# train our model 
print("Gradient descent start for b{0} and m{1} : Error{2}".format(initial_b,initial_m,compute_error(initial_b,initial_m,points)))
[b,m] = Gradient_descent_runner(points,initial_b,initial_m,learning_rate,num_iteration)
print("Gradient descent end for b{1} and m{2} : Error{3}".format(num_iteration,b,m,compute_error(b,m,points)))



