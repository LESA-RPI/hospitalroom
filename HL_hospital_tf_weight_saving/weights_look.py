import numpy as np
import pickle


f = open('weights_block.txt', 'rb')
w = pickle.load(f)
weights=list(w[0][0])
bias=list(w[1][0])
print(len(weights))
print(bias)

#data=[1214, 2752, 2790, 1901, 1687, 1875, 2531, 2631, 2614, 1270, 2439, 2455, 1649, 1733, 1744, 1728, 1724, 1727]
#layer= np.matmul(data,weights)+ bias
#label_estimate = np.argmax(layer)
#print(layer)
#print(label_estimate)

"""
[1x18]x[18x5] = [1x5] +bias
"""
# weights : 18x[1x5]
# bias: 1x5
