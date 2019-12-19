try:
    from Tkinter import *
except ImportError:
    from tkinter import *  # Python 3
import numpy as np
from block_class import *


global label
global all_data
global all_background
global num_frames


def one_line_format(frames):
    line = []
    for i in range(num_frames):
        line += frames[i]
    # print(len(line))
    return line


def merge_data(data0, data1):
    frames = []
    temp = [] # each elements is 9x2xnum_frames numbers
    for i in range(min(len(data0), len(data1))):
        temp_1x18 = []
        for j in range(9):
            temp_1x18.append(data0[i][j])
        for j in range(9):
            temp_1x18.append(data1[i][j])
        if len(frames) < num_frames:
            frames.insert(0, temp_1x18.copy())
            continue
        else:
            frames.pop()
            frames.insert(0, temp_1x18.copy())
        temp.append(one_line_format(frames))
    return temp


def add_data(temp_data, act_num, all_data, label):
    all_data += temp_data
    for i in range(len(temp_data)):
        label.append(act_num)


if __name__ == '__main__':
    label = []
    all_data = []
    num_frames = 8
    """
    since the old data is named by researcher name and do not have the falling activity, 
    u can change them to subject format and create an empty falling file to short this program.
    """
    # background
    all_background = merge_data(np.loadtxt("mo_background_0.txt"), np.loadtxt("mo_background_1.txt"))
    all_background += merge_data(np.loadtxt("daniel_background_0.txt"), np.loadtxt("daniel_background_1.txt"))
    all_background += merge_data(np.loadtxt("isaac_background_0.txt"), np.loadtxt("isaac_background_1.txt"))
    all_background += merge_data(np.loadtxt("ruijie_background_0.txt"), np.loadtxt("ruijie_background_1.txt"))
    all_background += merge_data(np.loadtxt("hh_background_0.txt"), np.loadtxt("hh_background_1.txt"))
    i = 1
    while i < 4 :
        all_background += merge_data(np.loadtxt("subject"+str(i)+"_background_0.txt"), np.loadtxt("subject"+str(i)+"_background_1.txt"))
        i += 1
    np.savetxt("all_background_block.txt", all_background)
    # activity 1
    add_data(merge_data(np.loadtxt("mo_1_0.txt"), np.loadtxt("mo_1_1.txt")), 1, all_data, label)
    add_data(merge_data(np.loadtxt("daniel_1_0.txt"), np.loadtxt("daniel_1_1.txt")), 1, all_data, label)
    add_data(merge_data(np.loadtxt("isaac_1_0.txt"), np.loadtxt("isaac_1_1.txt")), 1, all_data, label)
    add_data(merge_data(np.loadtxt("ruijie_1_0.txt"), np.loadtxt("ruijie_1_1.txt")), 1, all_data, label)
    add_data(merge_data(np.loadtxt("hh_1_0.txt"), np.loadtxt("hh_1_1.txt")), 1, all_data, label)
    # activity 2
    add_data(merge_data(np.loadtxt("mo_2_0.txt"), np.loadtxt("mo_2_1.txt")), 2, all_data, label)
    add_data(merge_data(np.loadtxt("daniel_2_0.txt"), np.loadtxt("daniel_2_1.txt")), 2, all_data, label)
    add_data(merge_data(np.loadtxt("isaac_2_0.txt"), np.loadtxt("isaac_2_1.txt")), 2, all_data, label)
    add_data(merge_data(np.loadtxt("ruijie_2_0.txt"), np.loadtxt("ruijie_2_1.txt")), 2, all_data, label)
    add_data(merge_data(np.loadtxt("hh_2_0.txt"), np.loadtxt("hh_2_1.txt")), 2, all_data, label)

    """activity 3,4,5 labeled as one activity because they are too close to recognize by 3x3 pods"""
    # activity 3
    add_data(merge_data(np.loadtxt("mo_3_0.txt"), np.loadtxt("mo_3_1.txt")), 3, all_data, label)
    add_data(merge_data(np.loadtxt("daniel_3_0.txt"), np.loadtxt("daniel_3_1.txt")), 3, all_data, label)
    add_data(merge_data(np.loadtxt("isaac_3_0.txt"), np.loadtxt("isaac_3_1.txt")), 3, all_data, label)
    add_data(merge_data(np.loadtxt("ruijie_3_0.txt"), np.loadtxt("ruijie_3_1.txt")), 3, all_data, label)
    add_data(merge_data(np.loadtxt("hh_3_0.txt"), np.loadtxt("hh_3_1.txt")), 3, all_data, label)
    # activity 4
    add_data(merge_data(np.loadtxt("mo_4_0.txt"), np.loadtxt("mo_4_1.txt")), 3, all_data, label)
    add_data(merge_data(np.loadtxt("daniel_4_0.txt"), np.loadtxt("daniel_4_1.txt")), 3, all_data, label)
    add_data(merge_data(np.loadtxt("isaac_4_0.txt"), np.loadtxt("isaac_4_1.txt")), 3, all_data, label)
    add_data(merge_data(np.loadtxt("ruijie_4_0.txt"), np.loadtxt("ruijie_4_1.txt")), 3, all_data, label)
    add_data(merge_data(np.loadtxt("hh_4_0.txt"), np.loadtxt("hh_4_1.txt")), 3, all_data, label)
    # activity 5
    add_data(merge_data(np.loadtxt("mo_5_0.txt"), np.loadtxt("mo_5_1.txt")), 3, all_data, label)
    add_data(merge_data(np.loadtxt("daniel_5_0.txt"), np.loadtxt("daniel_5_1.txt")), 3, all_data, label)
    add_data(merge_data(np.loadtxt("isaac_5_0.txt"), np.loadtxt("isaac_5_1.txt")), 3, all_data, label)
    add_data(merge_data(np.loadtxt("ruijie_5_0.txt"), np.loadtxt("ruijie_5_1.txt")), 3, all_data, label)
    add_data(merge_data(np.loadtxt("hh_5_0.txt"), np.loadtxt("hh_5_1.txt")), 3, all_data, label)
    # activity 6
    add_data(merge_data(np.loadtxt("mo_6_0.txt"), np.loadtxt("mo_6_1.txt")), 4, all_data, label)
    add_data(merge_data(np.loadtxt("daniel_6_0.txt"), np.loadtxt("daniel_6_1.txt")), 4, all_data, label)
    add_data(merge_data(np.loadtxt("isaac_6_0.txt"), np.loadtxt("isaac_6_1.txt")), 4, all_data, label)
    add_data(merge_data(np.loadtxt("ruijie_6_0.txt"), np.loadtxt("ruijie_6_1.txt")), 4, all_data, label)
    add_data(merge_data(np.loadtxt("hh_6_0.txt"), np.loadtxt("hh_6_1.txt")), 4, all_data, label)

#%%
    # the normal loop for labeling
    i = 1
    while i < 4:
        j = 1
        while j < 8:
            if j<4: # 1,2,3
                add_data(merge_data(np.loadtxt("subject"+str(i)+"_"+str(j)+"_0.txt"), np.loadtxt("subject"+str(i)+"_"+str(j)+"_1.txt")), j, all_data, label)
            elif 4<=j <=5: #4,5
                add_data(merge_data(np.loadtxt("subject" + str(i) + "_" + str(j) + "_0.txt"),
                                    np.loadtxt("subject" + str(i) + "_" + str(j) + "_1.txt")), 3, all_data, label)
            else: #6,7
                add_data(merge_data(np.loadtxt("subject" + str(i) + "_" + str(j) + "_0.txt"),
                                    np.loadtxt("subject" + str(i) + "_" + str(j) + "_1.txt")), j-2, all_data, label)
            j+=1
        i+=1

    # label the valid data, u can change it with other better choice in the future.
    test_data = []
    test_label = []
    i = 1
    while i < 6:  # subjecttest_1_0
        add_data(merge_data(np.loadtxt("test\subjecttest_" + str(i) + "_0.txt"),
                            np.loadtxt("test\subjecttest_" + str(i) + "_1.txt")), i, test_data, test_label)
        i += 1

    add_data(merge_data(np.loadtxt("test\subjecttest_backgtound_0.txt"),
                        np.loadtxt("test\subjecttest_backgtound_1.txt")), 0, test_data, test_label)

    # output the labeled files
    np.savetxt("all_data_block_stable.txt", all_data)
    np.savetxt("label_block_stable.txt", label)

    np.savetxt("valid_data_block_stable.txt", test_data)
    np.savetxt("valid_label_block_stable.txt", test_label)