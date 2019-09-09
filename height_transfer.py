try:
    from Tkinter import *
except ImportError:
    from tkinter import *  # Python 3
import numpy as np

global P2F  # the height of pod to the floor
global P2F_reading  # the reading value from the sensor
global ratio
global base_line
global base_distance_0
global base_distance_1
'''
maybe dont use the absolute value directly for transferring
recognize the background as the actual measured value. like ground=0, bed= 0.77m
-------- based on the difference, determine the current height?
a little fake????
'''
# right now based on the absolute value, the performance is accepted but not good enoght
# it should use difference instead
# right now use the first line as standard, should change to the average num


# transfer the reading data to the actual distance. unit: m
def num_to_height(num):
    x = (P2F/P2F_reading)*num
    return round(x, 2)


# cos(30)=0.866
def transfer_30_degree(heat_map):
    temp = []
    difference = [(base_line[i]-heat_map[i]) for i in range(9)]
    for i in range(9):
        temp.append(round(num_to_height(int(difference[i]*ratio[i]))+base_distance_0[i], 2))
    temp.append(heat_map[9])
    return temp


# cos(15)=0.966
def transfer_15_degree(heat_map):
    temp = []
    difference = [(base_line[i]-heat_map[i]) for i in range(9)]
    for i in range(9):
        temp.append(round(num_to_height(int(difference[i]*ratio[i]))+base_distance_1[i], 2))
    temp.append(heat_map[9])
    return temp


if __name__ == '__main__':

    base_distance_0 = [0, 0, 0, 0.77, 0.77, 0.77, 0, 0, 0]
    base_distance_1 = [0, 0, 0, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77]
    P2F = 2.4
    filename = input("filename:")
    background_filename = input("background filename:")
    f_blank = open(background_filename + '.txt', 'r')
    data_blank = f_blank.read()
    data_blank = data_blank.strip().split('\n')
    for i in range(len(data_blank)):
        data_blank[i] = data_blank[i].strip().split(' ')
        for j in range(9):
            data_blank[i][j] = int(data_blank[i][j])
        data_blank[i][9] = float(data_blank[i][9])

    f = open(filename+".txt", 'r')
    data = f.read()
    data = data.strip().split('\n')
    for i in range(len(data)):
        data[i] = data[i].strip().split(' ')
        for j in range(9):
            data[i][j] = int(data[i][j])
        data[i][9] = float(data[i][9])
        # data[i] = sort_transfer(data[i])

    # sample line for define the P2F value
    # print(data_blank[0])
    sum_background = [0 for i in range(9)]
    for i in range(len(data_blank)):
        for j in range(9):
            sum_background[j] += data_blank[i][j]
    base_line = [int(sum_background[i]/len(data_blank)) for i in range(9)]
    P2F_reading = base_line[4]+775

    flag_30 = False
    flag_15 = True
    # get the ratio from the original data
    # since the noise is very small, we can ignore it when calculate the ratio here
    ratio = [0 for i in range(9)]

    for i in range(9):
        if flag_15:
            if i < 3:
                ratio[i] = round(P2F_reading / base_line[i], 2)
            else:
                ratio[i] = round(base_line[4] / base_line[i], 2)
        elif flag_30:
            if 3 <= i <= 5:
                ratio[i] = round(base_line[4] / base_line[i], 2)
            else:
                ratio[i] = round(P2F_reading / base_line[i], 2)
    print(ratio)

    for i in range(len(data)):
        if flag_30:
            data[i] = transfer_30_degree(data[i])
        elif flag_15:
            data[i] = transfer_15_degree(data[i])

    f = open(filename+"_height.txt", 'w')
    for i in range(len(data)):
        for j in range(len(data[i])):
            f.write(str(data[i][j])+' ')
        f.write('\n')
    f.close()







