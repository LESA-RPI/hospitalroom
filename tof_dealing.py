import os
import numpy as np
global heat_map


# row and column should be reverse in these 3 algo
# 20x25 -> 8x8, 10x10
def transfer8x8(tof_map):
    temp_map = [[0 for j in range(8)] for i in range(8)]
    i, j = 0, 0
    x, y = 0, 0
    while i < 20 and x < 10:
        if j == 21:
            temp_map[x][y] = minimum(tof_map, 3, 4, i, j)
        else:
            temp_map[x][y] = minimum(tof_map, 3, 3, i, j)
        j += 3
        if j == 24:
            j = 0
            y = 0
            x += 1
            if i % 5 == 0:
                i += 2
            elif i % 5 == 2:
                i += 3
            else:
                print("row value wrong")
        else:
            y += 1
    return temp_map


def transfer10x10(tof_map):
    temp_map = [[0 for j in range(10)] for i in range(10)]
    i, j = 0, 0
    x, y = 0, 0
    while i < 20 and x < 10:
        temp_map[x][y] = minimum(tof_map, 2, 3, i, j)
        if j % 5 == 0:
            j += 2
            y += 1
        elif j % 5 == 2 and j != 22:
            j += 3
            y += 1
        elif j == 22:
            x += 1
            i += 2
            y = 0
            j = 0
        else:
            print("col value wrong")
    return temp_map


def transfer_4x4(tof_map):
    temp_map = [[0 for j in range(4)] for i in range(4)]
    i, j = 0, 0
    x, y = 0, 0
    while i < 20 and x < 4:
        if j == 18:
            temp_map[x][y] = minimum(tof_map, 5, 7, i, j)
            y = 0
            j = 0
            x += 1
            i += 5
        else:
            temp_map[x][y] = minimum(tof_map, 5, 6, i, j)
            y += 1
            j += 6
    return temp_map


def minimum(data_map, n_row, n_col, start_i, start_j):
    num = 10000
    for i in range(n_row):
        for j in range(n_col):
            if data_map[start_i+i][start_j+j] < num:
                num = data_map[start_i+i][start_j+j]
    # return the minimum num in this area
    return num


def one_line_format(matrix):
    temp = []
    for i in range(len(matrix)):
        temp += matrix[i]
    return temp


# the average value function
# if wanna use average way, replace the minimum with avg_of_area in
# functions transfer_4x4, transfer_8x8, transfer_10x10
def avg_of_area(data_map, n_row, n_col, start_i, start_j):
    sum_of_area = 0
    for i in range(n_row):
        for j in range(n_col):
            sum_of_area += data_map[start_i+i][start_j+j]
    avg = int(sum_of_area/(n_row*n_col))
    return avg


if __name__ == '__main__':
    filename_tof = input("filename_tof:")  # subject+'_'+ ac_num + '_tof'
    f = open(filename_tof + '.txt', 'r')
    d_tof = f.read()
    data_tof = d_tof.strip().split('\n')
    for i in range(len(data_tof)):
        data_tof[i] = data_tof[i].strip().split(' ')
        for j in range(500):
            data_tof[i][j] = int(data_tof[i][j])
        data_tof[i][500] = round(float(data_tof[i][500]), 4)
    line_num = 0
    matrix_4x4 = []
    matrix_8x8 = []
    matrix_10x10 = []
    while line_num < len(data_tof):
        temp_map = [[0 for j in range(25)] for i in range(20)]
        for i in range(20):
            for j in range(25):
                temp_map[i][j] = data_tof[line_num][i*25+j]
        matrix_4x4.append(one_line_format(transfer_4x4(temp_map)))
        matrix_8x8.append(one_line_format(transfer8x8(temp_map)))
        matrix_10x10.append(one_line_format(transfer10x10(temp_map)))
        line_num += 1
    # print(matrix_4x4)
    # C:\Users\DRJ_RPI\Google Drive\LESA\2019summer\tof_transferfile
    np.savetxt(os.path.join("tof_transferfile", filename_tof+"_4x4.txt"), matrix_4x4, fmt='%i')
    np.savetxt(os.path.join("tof_transferfile", filename_tof+"_8x8.txt"), matrix_8x8, fmt='%i')
    np.savetxt(os.path.join("tof_transferfile", filename_tof + "_10x10.txt"), matrix_10x10, fmt='%i')





