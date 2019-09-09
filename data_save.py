"""
this program is used for saving data. cannot run with real time visualization(hospitalmap.py)
it requires same gateway setting up as hospitalmap.py
there should be 4 output file: 3 txt file for 2 pods and 20x25 matrix and 1 video file for WebCam
the name of 4 files should follow the naming rules in Readme.
"""

import numpy as np
import time
import timeit
import os
import json
import cv2


def check_same(array1, array2):
    for i in range(len(array1)):
        for j in range(len(array1[i])):
            if array1[i][j] != array2[i][j]:
                return False
    return True


def sort_map(orig_map):
    # the sort way is different from visualizing way
    # this is the original version, real time visualization requires corresponding with observation angle
    temp_list=[[0,0,0],[0,0,0],[0,0,0]]
    temp_list[1][2] = orig_map[0]
    temp_list[2][2] = orig_map[1]
    temp_list[2][1] = orig_map[2]
    temp_list[0][2] = orig_map[3]
    temp_list[0][1] = orig_map[4]
    temp_list[1][1] = orig_map[5]
    temp_list[0][0] = orig_map[6]
    temp_list[2][0] = orig_map[7]
    temp_list[1][0] = orig_map[8]
    return temp_list


if __name__ == '__main__':
    filename = input("filename1:")
    filename_flat = input("filename2:")
    filename_tof = input("filename TOF:")
    filename_camera = input("filename camera:")
    period = float(input("period:"))
    # video record initialization
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename_camera + '.avi', fourcc, 20.2, (640, 480))
    # first camera read will cause some lag?
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    f = open(filename + ".txt", 'w')  # only save the distance data now
    f2 = open(filename_flat + ".txt", 'w')  ###
    f3 = open(filename_tof + ".txt", 'w')

    with open('ArpaE.9pixel/data/all_data.json') as json_file:
        file1 = json.load(json_file)

    data_d = sort_map(file1['arpae/192.168.0.101']['distances'])
    data_d_flat = sort_map(file1['arpae/192.168.0.102']['distances'])

    # tof data: 25x20
    data_file = open('scr.tof_control/SCR/output-0.txt', 'r')
    data_TOF = data_file.read().strip()
    data_tof = data_TOF.split('\n')
    start_time = prev_time = prev_time_flat = prev_time_tof = timeit.default_timer()
    current_time = timeit.default_timer()
    for i in range(25):
        data_tof[i] = (data_tof[i].strip()).split('\t')
    for i in range(25):
        for j in range(20):
            data_tof[i][j] = int(data_tof[i][j])

    for i in range(3):
        for j in range(3):
            f.write(str(data_d[i][j]) + ' ')
            f2.write(str(data_d_flat[i][j]) + ' ')

    for i in range(25):
        for j in range(20):
            f3.write(str(data_tof[i][j]) + ' ')

    temp_time = float(current_time - start_time)
    f.write(str(temp_time) + '\n')
    f2.write(str(temp_time) + '\n')
    f3.write(str(temp_time) + '\n')
    prev_data = data_d.copy()
    prev_data_flat = data_d_flat.copy()
    prev_data_tof = data_tof.copy()

    while current_time - start_time < period:
        current_time = timeit.default_timer()
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        out.write(frame)
        read_flag = True
        try:
            with open('ArpaE.9pixel/data/all_data.json') as json_file:
                file1 = json.load(json_file)
            data_d = sort_map(file1['arpae/192.168.0.101']['distances'])
            data_d_flat = sort_map(file1['arpae/192.168.0.102']['distances'])

            data_file = open('scr.tof_control/SCR/output-0.txt', 'r')
            data_TOF = data_file.read().strip()
            data_tof = data_TOF.split('\n')
            for i in range(25):
                data_tof[i] = (data_tof[i].strip()).split('\t')
            for i in range(25):
                for j in range(20):
                    data_tof[i][j] = int(data_tof[i][j])
        except IndexError:
            print("TOF sensor not finish writing")
            read_flag = False
        except:
            print("cannot load")

        if current_time - prev_time >= 0.1:
            if not check_same(prev_data, data_d):
                for i in range(3):
                    for j in range(3):
                        f.write(str(data_d[i][j]) + ' ')
                temp_time = float(current_time - start_time)
                f.write(str(temp_time) + '\n')
                prev_time = timeit.default_timer()
                prev_data = data_d.copy()

        if current_time - prev_time_flat >= 0.1:
            if not check_same(prev_data_flat, data_d_flat):
                for i in range(3):
                    for j in range(3):
                        f2.write(str(data_d_flat[i][j]) + ' ')
                time_flat = float(current_time - start_time)
                f2.write(str(time_flat) + '\n')
                prev_time_flat = timeit.default_timer()
                prev_data_flat = data_d_flat.copy()

        if current_time - prev_time_tof >= 0.1 and read_flag:
            if not check_same(prev_data_tof, data_tof):
                for i in range(25):
                    for j in range(20):
                        f3.write(str(data_tof[i][j]) + ' ')
                time_tof = float(current_time - start_time)
                f3.write(str(time_tof)+'\n')
                prev_time_tof = timeit.default_timer()
                prev_data_tof = data_tof.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()

    f.close()
    f2.close()
    f3.close()
    print("Done.")
    print(time.ctime())
    cv2.destroyAllWindows()
'''
file f and f2 record multiple times of one 9-pixel sensor's data
sensor0 1 2 3 4 5 6 7 8 time

file f3 record multiple times of the 20x25 tof cam sensor
0 1 2 3 4 5....499 time
'''
