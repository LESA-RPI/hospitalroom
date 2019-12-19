import numpy as np
import pickle

global weights
global bias


def action_num(data1x144):
    layer = np.matmul(data1x144, weights) + bias
    label_estimate = np.argmax(layer)
    return label_estimate


valid_data = np.loadtxt('valid_data_block_stable.txt')
valid_label = np.loadtxt('valid_label_block_stable.txt')

# print(valid_data[0])
# print(valid_label[0])

f = open('weights_8b_stable.txt', 'rb')  # open the output file of hospital_save_weight_v0.py
w = pickle.load(f)
weights = list(w[0][0])
bias = list(w[1][0])

count = [[0 for i in range(6)] for j in range(6)]
sum_count = [0 for i in range(6)]

for i in range(len(valid_data)):
    dec_num = action_num(valid_data[i])
    sum_count[int(valid_label[i])] += 1
    count[int(valid_label[i])][dec_num] += 1

print(count)
print(sum_count)



