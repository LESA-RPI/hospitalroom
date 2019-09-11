# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:10:57 2019

@author: luh6r
"""

import numpy as np
import tensorflow as tf
import pickle

'''
    loading files

    input: 
        all_backtround.txt  -> background
        all_data.txt        -> data
        label.txt           -> label
    labels: 
        0-> stand on corner
        1-> sit on the middle of bed, as patient
        2-> lie down on bed, middle
        3-> lie down on bed, left
        4-> lie down on bed, right
        5-> seat on the side of bed, as visitor
        6-> background 
    output:
        label_train     -> training label
        data_train      -> training data

        data_valid      -> valid data
        label_valid     -> valid label
'''

all_background = np.loadtxt('all_background.txt')
all_data = np.loadtxt('all_data.txt')
all_label = np.loadtxt('label.txt')
all_label = np.array(all_label-1,dtype=np.int32) 
label_background = np.zeros(all_background.shape[0],dtype=np.int32)+6
data = np.vstack((all_data,all_background))
label = np.hstack((all_label,label_background))
idx = np.random.choice(data.shape[0],size=data.shape[0],replace=False)                
data = data[idx,:]
label = label[idx]

breakpoint = int(0.8*data.shape[0])
data_train = data[:breakpoint]
label_train = label[:breakpoint]

data_valid = data[breakpoint:]
label_valid = label[breakpoint:] 

#%%
'''
    define Neural network graph

    structure:
        1. 1 layer
        2. 18 input notes
        3. 7 output classes
'''
class_num = 7
tf.reset_default_graph()
data_placeholder = tf.placeholder(tf.float32, shape=[None,all_data.shape[1]])
label_placeholder = tf.placeholder(tf.int64, shape=[None])

x = data_placeholder
y = label_placeholder

layer1 = tf.layers.dense(inputs=x,units=class_num,use_bias = True,activation=None, name="layer1")

# collect weights
with tf.variable_scope('layer1',reuse = True):
    w = tf.get_variable('kernel')
    b = tf.get_variable('bias')
outputs = layer1
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = outputs)

optimizer = tf.train.AdamOptimizer().minimize(loss)

prediction = tf.argmax(outputs,1)
equality = tf.equal(prediction,y)
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

init_op = tf.global_variables_initializer()

confusion_matrix_reg = np.zeros([class_num,class_num])
confusion_matrix_train =  np.zeros([class_num,class_num])
confusion_matrix_valid = np.zeros([class_num,class_num])
label_train_sum = np.zeros(class_num)
label_valid_sum = np.zeros(class_num)
valid_accuracy = []

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(200000):
        train_acc_sum = 0
        valid_acc_sum = 0
        label_train_sum = np.zeros(class_num)
        label_valid_sum = np.zeros(class_num)
        confusion_matrix_reg = np.zeros([class_num,class_num])
        
        pre,_,acc = sess.run([prediction,optimizer,accuracy],feed_dict={data_placeholder:data_train, label_placeholder: label_train})
        for k in range(label_train.shape[0]):
            label_train_sum[label_train[k]] += 1
            confusion_matrix_reg[pre[k],label_train[k]] += 1
        confusion_matrix_train = confusion_matrix_reg/label_train_sum
        print("Epoch: {}, train_accuracy:{}".format(i,acc)) 
        
        confusion_matrix_reg = np.zeros([class_num,class_num])
        bias,weight,pre,acc = sess.run([b,w,prediction,accuracy],feed_dict={data_placeholder:data_valid, label_placeholder: label_valid})
        for k in range(label_valid.shape[0]):
            label_valid_sum[label_valid[k]] += 1
            confusion_matrix_reg[pre[k],label_valid[k]] += 1
        confusion_matrix_valid = confusion_matrix_reg/label_valid_sum
        print("Epoch: {}, valid_accuracy:{}".format(i,acc)) 

weight = np.array(weight)
bias = np.array(bias)

f = open('weights.txt','wb')
pickle.dump([(weight,'weights'),(bias,'bias')],f)
f.close()