"""
Two ideas for making decisions:
    1. output all single frames' result
        if all of them same: stable action state
        if different
            a.action1 -> action2 ?
            b.cannot make decision ---- unknown
    2. consider the whole frames as a block (familiar with smart conference room)
        requires new training for the algo ---- labelling as the Frames class structure

Right now the second idea has better performance by training accuracy. Both of them needs to be tested by building a test case.
"""
import numpy as np
import json
import pickle
import math

#%%


class Frames:

    def __init__(self,w,b):
        self.frames = []
        self.actions = []  # if use ideas2 it would be an int instead of a list
        self.weights = list(w)  # matrix from ML algo
        self.bias = list(b)  # matrix from ML algo

    # add frame when recognize a new frame(this include noise difference)
    def add_frame(self, temp_map,action_id):
        if len(self.frames) == 8:    # vary with the performance, 8 is my first guess
            # delete the last subject which is the oldest one
            self.frames.pop()
            # insert the new frame at the beginning
            self.frames.insert(0, temp_map.copy())
            self.actions.pop()
            self.actions.insert(0, action_id)
        else:
            if len(self.frames) < 8:
                self.frames.insert(0, temp_map.copy())
                self.actions.insert(0, action_id)
            else:  # this is not right, something goes wrong
                print("the frames class too long")
                while len(self.frames) > 8:
                    self.frames.pop()
                    self.actions.pop()

    # get the average value for current 5 frames
    def check_frame_different(self, temp_map):
        # print(self.frames[0])
        for i in range(18):
            if self.frames[0][i] != temp_map[i]:
                return True
        return False

    def get_length(self):
        return len(self.frames),len(self.actions)

    def action_nums(self):
        for i in range(8):
            data1x18 = self.frames[i]
            layer = np.matmul(data1x18, self.weights) + self.bias
            label_estimate = np.argmax(layer)
            self.actions[i] = int(label_estimate)

    def check_all_same(self):
        # print(len(self.actions))
        for i in range(len(self.actions)-1):
            if self.actions[i] != self.actions[i+1]:
                return False
        return True

    # layer = np.matmul(data1x18, weights) + bias
    def percentage_cal(self):
        dec = list(np.matmul(self.frames[0], self.weights) + self.bias)
        dec_exp = [math.exp(dec[i]) for i in range(len(dec))]
        sum_exp = sum(dec_exp)
        return math.exp(dec_exp[self.actions[0]])/sum_exp


    # make decision based on the actions list (single frames)
    def decision(self):
        if self.check_all_same():
            return self.actions[0],1
        # else: percentage
        return self.actions[0]  # ,self.percentage_cal()




