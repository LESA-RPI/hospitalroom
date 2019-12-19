try:
    from Tkinter import *
except ImportError:
    from tkinter import *  # Python 3
import numpy as np

global label
global all_data
global all_background


def merge_data(data0, data1):
    temp = []
    for i in range(min(len(data0), len(data1))):
        temp.append([])
        for j in range(9):
            temp[i].append(data0[i][j])
        for j in range(9):
            temp[i].append(data1[i][j])
    return temp


def add_data(temp_data, act_num, all_data, label):
    all_data += temp_data
    for i in range(len(temp_data)):
        label.append(act_num)


if __name__ == '__main__':
    label = []
    all_data = []
    # print(np.loadtxt("mo_background_0.txt")[0])
    # background
    all_background = merge_data(np.loadtxt("mo_background_0.txt"), np.loadtxt("mo_background_1.txt"))
    all_background += merge_data(np.loadtxt("daniel_background_0.txt"), np.loadtxt("daniel_background_1.txt"))
    all_background += merge_data(np.loadtxt("isaac_background_0.txt"), np.loadtxt("isaac_background_1.txt"))
    all_background += merge_data(np.loadtxt("ruijie_background_0.txt"), np.loadtxt("ruijie_background_1.txt"))
    all_background += merge_data(np.loadtxt("hh_background_0.txt"), np.loadtxt("hh_background_1.txt"))

    i = 1
    while i < 4 : #4
        all_background += merge_data(np.loadtxt("subject"+str(i)+"_background_0.txt"),
                                     np.loadtxt("subject"+str(i)+"_background_1.txt"))
        i += 1
    np.savetxt("all_background.txt", all_background)
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
    i = 1
    while i < 4: #4
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

    np.savetxt("all_data.txt", all_data)
    np.savetxt("label.txt", label)

    test_data = []
    test_label = []
    i=1
    while i < 6: # subjecttest_1_0
        add_data(merge_data(np.loadtxt("test\subjecttest_" + str(i) +"_0.txt"),
                            np.loadtxt("test\subjecttest_" + str(i) +"_1.txt")), i, test_data, test_label)
        i+=1

    add_data(merge_data(np.loadtxt("test\subjecttest_backgtound_0.txt"),
                        np.loadtxt("test\subjecttest_backgtound_1.txt")), 0, test_data, test_label)
    np.savetxt("valid_data.txt", test_data)
    np.savetxt("valid_label.txt", test_label)

