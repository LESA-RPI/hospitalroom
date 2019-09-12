'''
Hao Lu
2019/09/12 

Description:
    this file is used to do the forward propagation of one layer neural network (logistic regression) 
'''

import numpy as np 
import pickle
import os

os.chdir(r'C:\Users\Admin\Documents\GitHub\hospitalroom\HL_hospital_tf_weight_saving')
'''
    read files: input weights, bias and raw data
'''
f = open('weights.txt','rb')
Weights = pickle.load(f)
f.close()

W = Weights[0][0]
b = Weights[1][0]

all_background = np.loadtxt('all_background.txt')
all_data = np.loadtxt('all_data.txt')
data = np.vstack((all_data,all_background))

#%%
'''
    forword propagation:
    1. math theory:
        output = data*W+bias


    data : n*18
    W    : 18*7
    bias : 7
'''
def layer1(x, Weights, biases):
    '''
        logistic regression layer
        18->7
        activation function: softmax
    '''
    Wx_plus_b = np.matmul(x,Weights) + biases
    return Wx_plus_b

#def softmax(x):
#    '''
#        compute softmax values for each sets of scores in x
#    '''
#    x_exp = np.exp(x)
#    x_exp_sum = np.reshape(np.sum(np.exp(x),axis = 1),[-1,1])
#    return x_exp/x_exp_sum


outputs = layer1(data,W,b)

lable_estimate = np.argmax(outputs,axis = 1)