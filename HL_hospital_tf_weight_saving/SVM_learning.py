# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:13:54 2019

@author: Admin
"""

from sklearn import svm
import pickle
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,confusion_matrix
"""
with open('label_block.txt', 'rb') as f:
    label_all = pickle.load(f)
with open('all_data_block.txt', 'rb') as f:
    data_all = pickle.load(f)
"""

data_all = np.loadtxt('all_data.txt')
label_all = np.loadtxt('label.txt')
    
pca = PCA(n_components=18)
data_training = pca.fit_transform(data_all)

break_point = int(0.9*data_all.shape[0])
data_train = data_training[:break_point]
label_train = label_all[:break_point]

data_valid = data_training[break_point:]
label_valid = label_all[break_point:] 

model = svm.SVC(kernel='rbf', C=2, gamma=0.00000165)

print( "Training model.")
#train model
model.fit(data_train, label_train)

predicted_labels = model.predict(data_valid)

print( "FINISHED classifying. accuracy score : ", end='')
acc = accuracy_score(label_valid, predicted_labels)
con_matrix = confusion_matrix(label_valid,predicted_labels)
print( accuracy_score(label_valid, predicted_labels))
print(con_matrix)
