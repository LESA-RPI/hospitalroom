try:
    from Tkinter import *
except ImportError:
    from tkinter import *  # Python 3
import numpy as np
import time
import timeit

global heat_map
global blank_map
global now_map
global recent_5_frame
global heat_map_flat
global blank_map_flat
global now_map_flat
global recent_5_frame_flat

# 3x3 matrix for all global variables
heat_map = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
blank_map = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
now_map = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
heat_map_flat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
blank_map_flat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
now_map_flat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


# a class which record the past 5 frames(4-5 seconds), to recognize the new blank frame/map
class Frames:

    def __init__(self, m1, m2):
        # each frame is [3x3 map, recorded time]: now the map is array, could be tranferred to list if needed
        # these frames could be organized as a list in the future for more frames and longer time
        self.frame0 = [None, None]
        self.frame1 = [None, None]
        self.frame2 = [None, None]
        self.frame3 = [None, None]
        self.frame4 = [None, None]
        self.blank = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.frame0[0] = m1.copy()
        self.frame0[1] = timeit.default_timer()
        self.frame1[0] = m1.copy()
        self.frame2[0] = m1.copy()
        self.frame3[0] = m1.copy()
        self.frame4[0] = m1.copy()
        self.blank = m2.copy()

    # add frame when recognize a new frame(this include noise difference)
    def add_frame(self, temp_map):
        self.frame4[0] = self.frame3[0].copy()
        if self.frame3[1] is not None : self.frame4[1] = self.frame3[1]

        self.frame3[0] = self.frame2[0].copy()
        if self.frame2[1] is not None : self.frame3[1] = self.frame2[1]

        self.frame2[0] = self.frame1[0].copy()
        if self.frame1[1] is not None : self.frame2[1] = self.frame1[1]

        self.frame1[0] = self.frame0[0].copy()
        self.frame1[1] = self.frame0[1]

        self.frame0[0] = temp_map.copy()
        self.frame0[1] = timeit.default_timer()

    # get the average value for current 5 frames
    def avg_frame(self):
        avg = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(3):
            for j in range(3):
                avg[i][j] = int((self.frame0[0][i][j] + self.frame1[0][i][j] + self.frame2[0][i][j] +
                                 self.frame3[0][i][j] + self.frame4[0][i][j]) / 5)
        return avg

    # check if the 5 frames are same(the difference between each other is in noise area)
    # if all 5 frames are same, then set the average of these 5 frames as new blank frame
    def check_new_blank(self):
        # global blank_map
        # check if still the original blank frame
        if error_area(self.frame0[0], self.blank):
            return False
        # check if stable
        temp = self.avg_frame()
        if not (error_area(self.frame0[0], temp)): return False
        if not (error_area(self.frame1[0], temp)): return False
        if not (error_area(self.frame2[0], temp)): return False
        if not (error_area(self.frame3[0], temp)): return False
        if not (error_area(self.frame4[0], temp)): return False

        # if stable, change the blank map to the avg_frame
        return True

    # set new blank frame in class
    def set_blank(self, temp_m):
        self.blank = temp_m.copy()

    # print the 5 frames
    def print(self):
        print(self.frame0)
        print(self.frame1)
        print(self.frame2)
        print(self.frame3)
        print(self.frame4)


# check if the two map frames are same, in noise area
def error_area(m1, m2):  # check if noise
    for i in range(3):
        for j in range(3):
            if abs(m1[i][j] - m2[i][j]) > 30:
                return False
    return True  # true= noise, same


# heat map color statement
def pseudocolor(value, minval, maxval, palette):
    """ Maps given value to a linearly interpolated palette color. """
    max_index = len(palette) - 1
    # Convert value in range minval...maxval to the range 0..max_index.
    v = (float(value - minval) / (maxval - minval)) * max_index
    i = int(v)
    f = v - i  # Split into integer and fractional portions.
    c0r, c0g, c0b = palette[i]
    c1r, c1g, c1b = palette[min(i + 1, max_index)]
    dr, dg, db = c1r - c0r, c1g - c0g, c1b - c0b
    return c0r + (f * dr), c0g + (f * dg), c0b + (f * db)  # Linear interpolation.


def colorize(value, minval, maxval, palette):
    """ Convert value to heatmap color and convert it to tkinter color. """
    color = (int(c * 255) for c in pseudocolor(value, minval, maxval, palette))
    return '#{:02x}{:02x}{:02x}'.format(*color)  # Convert to hex string.


def heatmap():
    global now_map
    global heat_map
    global heat_map_flat
    global now_map_flat
    canvas1.delete('all')
    canvas2.delete('all')
    # canvas3.delete('all')
    try:
        MAP = np.load('distances.npy')
    except:
        return None

    # 0 is the 30 degree data, 1 is the 15 degree data
    now_map = MAP[0]
    now_map_flat = MAP[1]
    # get the difference between current data and blank frame: blank data - current data
    for i in range(3):
        for j in range(3):
            heat_map[i][j] = blank_map[i][j] - now_map[i][j]  # data could be negative
            heat_map_flat[i][j] = blank_map_flat[i][j] - now_map_flat[i][j]

    # draw the heat map. the blank data is the mean value.
    # if the stuff is closer to the sensor, the color would be more green. this could be reversed if necessary
    for y, row in enumerate(heat_map):
        for x, temp in enumerate(row):
            x0, y0 = x * rect_width, y * rect_height
            x1, y1 = x0 + rect_width - border, y0 + rect_height - border
            color = colorize(2000 + temp, 0, 4000, palette)  # the base color is the mean value
            canvas1.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)

    for y, row in enumerate(heat_map_flat):
        for x, temp in enumerate(row):
            x0, y0 = x * rect_width, y * rect_height
            x1, y1 = x0 + rect_width - border, y0 + rect_height - border
            color = colorize(temp + 2000, 0, 4000, palette)
            canvas2.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)


# check if two map are totally same, noise difference are also different.
# this function is used for recognizing new frame
def check_same(m1, m2):
    for i in range(3):
        for j in range(3):
            if m1[i][j] != m2[i][j]: return False
    return True


# infinite loop
def main():
    global blank_map
    global now_map
    global blank_map_flat
    global now_map_flat
    global recent_5_frame
    global recent_5_frame_flat
    # declare the previous map as original one
    prev_map = now_map.copy()
    prev_map_flat = now_map_flat.copy()

    while 1:
        # updata the heat map
        heatmap()
        time.sleep(0.1)
        root.update()

        print("heat map:", heat_map)
        print("blank frame:", blank_map)
        # recent_5_frame.print()
        print("heat map flat:", heat_map_flat)
        print("blank frame flat:", blank_map_flat)

        # add the frame to 2 class variables if new frame is recognizeds
        if not check_same(now_map_flat, prev_map_flat):
            recent_5_frame_flat.add_frame(now_map_flat.copy())
            prev_map_flat = now_map_flat.copy()
        if not check_same(now_map, prev_map):
            print("now map:", now_map)
            recent_5_frame.add_frame(now_map.copy())
            prev_map = now_map.copy()

        # if stable situation, replace the blank map as current data
        if recent_5_frame.check_new_blank():
            blank_map = recent_5_frame.avg_frame().copy()
            recent_5_frame.set_blank(blank_map)
            np.save('blank_normal', blank_map)

        if recent_5_frame_flat.check_new_blank():
            blank_map_flat = recent_5_frame_flat.avg_frame().copy()
            recent_5_frame_flat.set_blank(blank_map_flat)
            np.save('blank_flat', blank_map_flat)


def OnExit():
    root.destroy()


# import data from files, the zero map for two sensors
f1 = input("blank filename:")
f2 = input("blank filename for map2:")

# sensor 1, 30 degree
blank_file = open(f1 + ".txt", 'r').read()
blank_file = blank_file.strip()
content = blank_file.split('\n')
r1 = []
for line in content:
    line = line.strip()
    data = line.split(' ')
    temp = []
    for i in range(9):
        temp.append(int(data[i]))
    r1.append(temp.copy())
sum_blank = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(r1)):
    for j in range(9):
        sum_blank[j] += r1[i][j]
avg_blank = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(9):
    avg_blank[i] = int(sum_blank[i] / len(r1))
for i in range(3):
    for j in range(3):
        blank_map[i][j] = avg_blank[i * 3 + j]

# sensor 2, 15 degree, flatten one
blank_file_flat = open(f2 + ".txt", 'r').read()
blank_file_flat = blank_file_flat.strip()
content_flat = blank_file_flat.split('\n')
r1_flat = []
for line in content_flat:
    line = line.strip()
    data = line.split(' ')
    temp = []
    for i in range(9):
        temp.append(int(data[i]))
    r1_flat.append(temp.copy())
sum_blank_flat = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(r1_flat)):
    for j in range(9):
        sum_blank_flat[j] += r1_flat[i][j]
avg_blank_flat = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(9):
    avg_blank_flat[i] = int(sum_blank_flat[i] / len(r1_flat))
for i in range(3):
    for j in range(3):
        blank_map_flat[i][j] = avg_blank_flat[i * 3 + j]
np.save('blank_normal', blank_map)
np.save('blank_flat', blank_map_flat)
# create the two class
recent_5_frame_flat = Frames(blank_map_flat, blank_map_flat)
recent_5_frame = Frames(blank_map, blank_map)
# recent_5_frame.print()
# recent_5_frame_flat.print()

# drawing the heat map
palette = (1, 0, 0), (1, 1, 0), (0, 1, 0)
root = Tk()
root.title('sub map')
# the area of frame is flexible, but the place location needs to change with it
frame = Frame(root, width=600, height=800)
frame.pack()
panel = Label(frame)

width, height = 300, 300  # Canvas size.
rows, cols = 3, 3
rect_width, rect_height = width // rows, height // cols
border = 0
canvas1 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas2 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas1.place(relx=0.25, rely=0.6)
canvas2.place(relx=0.25, rely=0.1)
panel.place(relx=0, rely=0)
root.resizable(width=False, height=False)
main()
