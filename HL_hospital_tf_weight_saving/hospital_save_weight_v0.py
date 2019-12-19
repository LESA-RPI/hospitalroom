# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:10:57 2019

@author: luh6r


edit from 7 act num to 5 act num, 3,4,5 merge to 3
author:Ruijie Du
"""


import numpy as np
import tensorflow as tf
import pickle
import time

'''
    loading files

    input: 
        all_backtround.txt  -> background
        all_data.txt        -> data
        label.txt           -> label
    labels: 
        0-> background 
        1-> stand on corner
        2-> sit on the middle of bed, as patient
        3-> lie down on bed, # middle
            # 2-> lie down on bed, left
            # 2-> lie down on bed, right
        4-> seat on the side of bed, as visitor
        5-> falling
    output:
        label_train     -> training label
        data_train      -> training data

        data_valid      -> valid data
        label_valid     -> valid label
'''

all_background = np.loadtxt('all_background_block.txt')
all_data = np.loadtxt('all_data_block_stable.txt')
all_label = np.loadtxt('label_block_stable.txt')
all_label = np.array(all_label,dtype=np.int32)
label_background = np.zeros(all_background.shape[0],dtype=np.int32)  #
data = np.vstack((all_data,all_background))
label = np.hstack((all_label,label_background))
label = np.array(label,dtype=np.int32)
# random the training data
idx =np.random.choice(data.shape[0],size=data.shape[0],replace=False)
data = data[idx,:]
label = label[idx]
breakpoint = int(data.shape[0])

# import test data
valid_data = np.loadtxt('valid_data_block_stable.txt')
valid_label = np.loadtxt('valid_label_block_stable.txt')
valid_label = np.array(valid_label, dtype=np.int32)
label = np.hstack((label,valid_label))
label = np.array(label,dtype=np.int32)
data = np.vstack((data,valid_data))

data_train = data[:breakpoint]
label_train = label[:breakpoint]

data_valid = data[breakpoint:]
label_valid = label[breakpoint:]
# print(data_train.shape,data_valid.shape)
#%%
'''
    define Neural network graph

    structure:
        1. 1 layer
        2. 18 input notes
        3. 6 output classes
'''
# tf sef up
class_num = 6  # the num of act and background
tf.compat.v1.reset_default_graph()
data_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None,all_data.shape[1]])
label_placeholder = tf.compat.v1.placeholder(tf.int64, shape=[None])

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
# [1 x num_act] -> e^(xxx) -> act_resulr/sum(e) ->
equality = tf.equal(prediction,y)
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

init_op = tf.global_variables_initializer()

confusion_matrix_reg = np.zeros([class_num,class_num])
confusion_matrix_train =  np.zeros([class_num,class_num])
confusion_matrix_valid = np.zeros([class_num,class_num])
label_train_sum = np.zeros(class_num)
label_valid_sum = np.zeros(class_num)
valid_accuracy = []

# training
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(150000):  # since the data is tiny, the overfit is serious, edit this number if necessary
        train_acc_sum = 0
        valid_acc_sum = 0
        label_train_sum = np.zeros(class_num)
        label_valid_sum = np.zeros(class_num)
        confusion_matrix_reg = np.zeros([class_num,class_num])
        
        pre,_,train_acc = sess.run([prediction,optimizer,accuracy],feed_dict={data_placeholder: data_train,
                                                                        label_placeholder: label_train})
        for k in range(label_train.shape[0]):
            label_train_sum[label_train[k]] += 1
            confusion_matrix_reg[pre[k],label_train[k]] += 1
        confusion_matrix_train = confusion_matrix_reg/label_train_sum
        print("Epoch: {}, train_accuracy:{}".format(i,train_acc))
        
        confusion_matrix_reg = np.zeros([class_num,class_num])
        bias,weight,pre,valid_acc = sess.run([b,w,prediction,accuracy],feed_dict={data_placeholder: data_valid,
                                                                            label_placeholder: label_valid})
        for k in range(label_valid.shape[0]):
            label_valid_sum[label_valid[k]] += 1
            confusion_matrix_reg[pre[k],label_valid[k]] += 1
        confusion_matrix_valid = confusion_matrix_reg/label_valid_sum
        print("Epoch: {}, valid_accuracy:{}".format(i,valid_acc))
        if train_acc == 1 and valid_acc == 1:
            break

# output the matrix
weight = np.array(weight)
bias = np.array(bias)

f = open('weights_8b_stable.txt','wb')
pickle.dump([(weight,'weights'),(bias,'bias')],f)
f.close()
