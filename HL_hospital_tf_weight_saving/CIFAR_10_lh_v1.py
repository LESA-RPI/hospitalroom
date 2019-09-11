# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:58:20 2019

@author: Hao Lu
"""

import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import skimage
import pandas as pd
#%%
########################################################################
'''
    various constants for the size of the images
'''
# Width and height of each image
img_size = 32

# Number of channels in each image, 3 channels: R G B
num_channels = 3

# Number of class.
num_classes = 10

# Number of training images
num_images_train = 50000

# Number of testing images
num_images_test = 5000
########################################################################
#%%

########################################################################
'''
    Build up dataset
'''
data_path = "C:\deeplearning\PA3\cifar10_v1\cifar_10_tf_train_test.pkl"

########### Load data #######################
data_file = open(data_path,"rb")
train_x, train_y, test_x, test_y = pickle.load(data_file, encoding="bytes")
data_file.close()
#
#plt.imshow(train_x[0])
#%%

############# Preprocessing for image data ############
# Converting to Floats and scaling
train_x = skimage.img_as_float32(train_x)
test_x = skimage.img_as_float32(test_x)

# Normalizing #(normalization for all channels) (?should I normalization for separate channels)
train_x_mean = np.mean(np.mean(train_x, axis = 1),axis = 1).reshape(-1,1,1,3)
test_x_mean = np.mean(np.mean(test_x, axis = 1),axis = 1).reshape(-1,1,1,3)
#test_x_mean = np.mean(test_x, axis = 0)
train_x_std = np.std(np.std(train_x, axis = 1),axis = 1).reshape(-1,1,1,3) 
test_x_std = np.std(np.std(test_x, axis =1 ),axis = 1).reshape(-1,1,1,3)
#train_x_std = np.std(train_x, axis = 0) 

train_x = (train_x-train_x_mean)
test_x = (test_x-test_x_mean)
train_y = np.array(train_y)
test_y = np.array(test_y)
#train_y = train_y.astype(np.int64)
#test_y = test_y.astype(np.int64)
#%%
# Create the training datasets
#dx_train = tf.data.Dataset.from_tensor_slices(train_x)
#dy_train = tf.data.Dataset.from_tensor_slices(train_y).map(lambda z: tf.one_hot(z,10))
#
## Zip x and y training data togather. And shuffle and batch
#train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).shuffle(500).repeat().batch(64)
#
## Creat the testing datasets
#dx_test = tf.data.Dataset.from_tensor_slices(test_x)
#dy_test = tf.data.Dataset.from_tensor_slices(test_y).map(lambda z: tf.one_hot(z,10))
#
## Zip x and y testing data togather
#test_dataset = tf.data.Dataset.zip((dx_test,dy_test)).shuffle(500).repeat().batch(30)
#
## Create general iterator
#iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
#next_element = iterator.get_next()
#
##print(train_dataset.output_types)
##print(train_dataset.output_shapes)
#
## Make datasets that we can initialize separately, but using the same structure via the common iterator
#training_init_op = iterator.make_initializer(train_dataset)
#testing_init_op = iterator.make_initializer(test_dataset)

##############  define weights ###############################
#%%
tf.reset_default_graph()

x = tf.placeholder("float32",[None,32,32,3])
y = tf.placeholder("int32",None)


label = tf.one_hot(y,num_classes)

# Creates Weights Xavier initialization

Weights = {
        'wc1': tf.get_variable('W1', shape=(5,5,3,32), initializer=tf.glorot_normal_initializer()),
        'wc2': tf.get_variable('W2', shape=(5,5,32,32), initializer=tf.glorot_normal_initializer()),
        'wc3': tf.get_variable('W3', shape=(3,3,32,64), initializer=tf.glorot_normal_initializer()),
        'wf' : tf.get_variable('Wf', shape=(576,10), initializer=tf.glorot_normal_initializer())}
# Creates biases Xavier initialization
biases = {
        'bc1':tf.get_variable('b1', shape=(32),initializer=tf.glorot_normal_initializer()),
        'bc2':tf.get_variable('b2', shape=(32),initializer=tf.glorot_normal_initializer()),
        'bc3':tf.get_variable('b3', shape=(64),initializer=tf.glorot_normal_initializer()),
        'bf': tf.get_variable('bf', shape=(10),initializer=tf.glorot_normal_initializer()),}

#############  define CNN model ###############
#%%
# Create convolution layer
def conv2d(x, W, b, strides=1):
    '''
        no padding, stride = 1
    '''
    # convolution operation
    layer_out = tf.nn.conv2d(
            x, 
            W,
            strides = [1, strides, strides, 1], 
            padding='VALID')
    
    # add bias
    layer_out = tf.nn.bias_add(layer_out,b)
    
    # active function
    return tf.nn.relu(layer_out)

# Create pooling layer
def pool2d(x, k=2):
    '''
        MAX pooling, strides = 2, window shape = 2, no padding
    '''
    layer_out = tf.nn.max_pool(
            x, 
            ksize = [1, k, k, 1], 
            strides = [1, k, k, 1],
            padding = 'VALID',
            )
    return layer_out

# Create fully connected layer
def full_cl(x, Weights, biases):
    '''
        576 -> 10, active function: softmax 
    '''
    Wx_plus_b1 = tf.matmul(x, Weights) + biases
    return tf.nn.softmax(Wx_plus_b1)
    
# Create CNN graph
def cnn_model(in_data, Weights, biases):
    '''
        3 convolution layers, 1 fully connective layer
    '''
    
    ## convolution layer 1
    cv_layer1 = conv2d(x = in_data, W = Weights['wc1'], b = biases['bc1'], strides = 1)
    cv_layer1 = pool2d(cv_layer1, 2)
    
    ## convolution layer 2
    cv_layer2 = conv2d(cv_layer1, Weights['wc2'], biases['bc2'])
    cv_layer2 = pool2d(cv_layer2, 2)
    
    ## conovolution layer 3 (no pooling layer)
    cv_layer3 = conv2d(cv_layer2, Weights['wc3'], biases['bc3'])
    
    ## flatten layer
    flat_layer = tf.reshape(cv_layer3,[-1,3*3*64])
    
    ## fully connective layer
    out = full_cl(flat_layer, Weights['wf'], biases['bf'])
    
    return out
##%%
################### define learning rate #####################
#learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
#
#def lr(epoch):
#    learning_rate = 1e-3
#    if epoch > 14:
#        learning_rate *=0.5e-3
#    elif epoch > 13:
#        learning_rate *=1e-3
#    elif epoch > 12:
#        learning_rate *= 1e-2
#    elif epoch > 10:
#        learning_rate *= 1e-2
#    return learning_rate
################## build up graph #########################
#%%
# predicet (forward propagation)
pred = cnn_model(x, Weights, biases)


# loss function( cross entropy)
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=pred))

# optimizer (backward propagation)

optimizer = tf.train.AdamOptimizer().minimize(loss)

#get accuracy
prediction = tf.argmax(pred, 1)
equality = tf.equal(prediction, tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
init_op = tf.global_variables_initializer()

#save model
#wc1=Weights['wc1']
#wc2=Weights['wc2']
#wc3=Weights['wc3']
#wf=Weights['wf']
#bc1=biases['bc1']
#bc2=biases['bc2']
#bc3=biases['bc3']
#bf=biases['bf']
tf.get_collection('validation_nodes')

tf.add_to_collection('validation_nodes',x)
tf.add_to_collection('validation_nodes',y)
tf.add_to_collection('validation_nodes',prediction)

saver = tf.train.Saver()
#%%
epochs = 20
bsize = 500
with tf.Session() as sess:
    sess.run(init_op)
    train_accuracy = []
    test_accuracy = []
#    sess.run(training_init_op)
    idx = np.random.choice(50000,size=50000)
    for j in range(epochs):

        for ii in range(100):
            
            tbatch = train_x[idx[ii*bsize:ii*bsize+bsize],:,:,:]
            lbatch = train_y[idx[ii*bsize:ii*bsize+bsize]]
            
            l,_,acc = sess.run([loss, optimizer, accuracy],
                               feed_dict={x:tbatch, y:lbatch})
            if ii % 50 == 0:
                print("Epoch: {},iterate{} loss: {:.3f}, training accuracy: {:.2f}%".format(j,ii,l,acc * 100))
    
##    sess.run(testing_init_op)
#    avg_acc = 0
#    for i in range(39):
#        acc = sess.run(accuracy)
#        avg_acc += acc
        test_acc = sess.run(accuracy,feed_dict = {x:test_x, y:test_y})
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Average validation set accuracy over {} epoch iterations is {:.2f}%".format(j,test_acc * 100)) 
    save_path = saver.save(sess,"./my_model")
    y_out = sess.run(equality,feed_dict = {x:test_x,y:test_y})
#    result_w = sess.run(wc1,wc2,wc3,wf)
#    result_b = sess.run(bc1,bc2,bc3,bf)
    W = sess.run(Weights)
    b = sess.run(biases)
#    test_label = sess.run(label, feed_dict = {y:test_y})
#%%    
acc_rate = [0,0,0,0,0,0,0,0,0,0]
acc_sum = [0,0,0,0,0,0,0,0,0,0]
for i in range(num_images_test):
    for j in range(10):
        if test_y[i] == j:
            acc_sum[j] += 1.
            if y_out[i] == 1:
                acc_rate[j] += 1.
for i in range(10):
    acc_rate[i] = acc_rate[i]/acc_sum[i]
#y_preds = pd.get_dummies(y_out.T.argmax(0))
#
#correct_matix = np.dot(test_label.T,y_preds)
#total_num_matrix = np.dot(test_label.T,test_label)
#
##a_rate_0 = correct_matix[0,0]/total_num_matrix[0,0]
##a_rate_1 = correct_matix[1,1]/total_num_matrix[1,1]
##a_rate_2 = correct_matix[2,2]/total_num_matrix[2,2]
##a_rate_3 = correct_matix[3,3]/total_num_matrix[3,3]
##a_rate_4 = correct_matix[4,4]/total_num_matrix[4,4]
#error_rate = []
#for i in range(10):
#    err_ratei = 1-correct_matix[i,i]/total_num_matrix[i,i]
#    error_rate.append(err_ratei)
#    print(err_ratei)
#    
fig = plt.figure(1)
plt.bar(range(0,10),acc_rate,0.4,color="blue")
plt.xlabel("Digit")
plt.ylabel("accuracy rate")
plt.title("Accuracy rate for each digit number (test data set)")

plt.savefig("acc_rate_digit.pdf")
    
filehandler = open("weights.txt","wb")
pickle.dump(W, filehandler)
filehandler.close()

filehandler = open("biases.txt","wb")
pickle.dump(b, filehandler)
filehandler.close()

filehandler = open("train_accuracy.txt","wb")
pickle.dump(train_accuracy, filehandler)
filehandler.close()

filehandler = open("test_accuracy.txt","wb")
pickle.dump(test_accuracy, filehandler)
filehandler.close()

plt.figure(2)
plt.plot(test_accuracy,color='red')    
plt.ylabel('accuracy rate')
plt.xlabel('iterations')   
plt.title('test accuracy rate vs iterations')   
plt.savefig('test_acc.pdf') 

plt.figure(3)
plt.plot(train_accuracy,color='red')    
plt.ylabel('accuracy rate')
plt.xlabel('iterations')   
plt.title('train accuracy rate vs iterations')   
plt.savefig('train_acc.pdf') 

filehandler = open("error_rate.txt","wb")
pickle.dump(error_rate, filehandler)
filehandler.close()

#%%
data_file = open("C:\deeplearning\PA3\cifar10_v1\weights.txt","rb")
W = pickle.load(data_file, encoding="bytes")
data_file.close()
result = W['wc1']
for j in range(32):
    plt.figure(j+3)
    Img = result[:,:,:,j]
    plt.imshow(Img)
    plt.colorbar()
    plt.title(str(j+1))

    name=str(j+1)+'.pdf'
    plt.savefig(name)