try:
    from Tkinter import *
except ImportError:
    from tkinter import *  # Python 3
import numpy as np
import time
import timeit

global heat_map0
global heat_map1
global heat_map_tof
global tof_map
global heat_max
global heat_min


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

def sort_transfer(orig_map):
    # sort the map in right consequence. Correspond to the observer view
    temp_list=[0,0,0,0,0,0,0,0,0,0]
    temp_list[0] = orig_map[2]
    temp_list[1] = orig_map[5]
    temp_list[2] = orig_map[8]
    temp_list[3] = orig_map[1]
    temp_list[4] = orig_map[4]
    temp_list[5] = orig_map[7]
    temp_list[6] = orig_map[0]
    temp_list[7] = orig_map[3]
    temp_list[8] = orig_map[6]
    temp_list[9] = orig_map[9]
    return temp_list

def heatmap():
    canvas1.delete('all')
    canvas2.delete('all')
    canvas3.delete('all')

    # 0-> mid one, 1 flatten one

    # print("heat map:\n", heat_map0)
    for y, row in enumerate(heat_map0):
        for x, temp in enumerate(row):
            x0, y0 = x * rect_width, y * rect_height
            x1, y1 = x0 + rect_width - border, y0 + rect_height - border
            color = colorize(temp, heat_min, heat_max, palette)
            canvas1.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)
    
    for y, row in enumerate(heat_map1):
        for x, temp in enumerate(row):
            x0, y0 = x * rect_width, y * rect_height
            x1, y1 = x0 + rect_width - border, y0 + rect_height - border
            color = colorize(temp, heat_min, heat_max, palette)
            canvas2.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)

    for y, row in enumerate(heat_map_tof):
        for x, temp in enumerate(row):
            x0, y0 = x * 20, y * 20 #20
            x1, y1 = x0 + 20 - border, y0 + 20 - border
            color = colorize(temp, heat_min, 4500, palette)
            canvas3.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)


def main(data0,data1,datatof):
    # sensor0/1 data is sorted in different way, needs to be transferred
    # tof data is not, needs to be transferred to right order
    data_0 = data0.strip().split('\n')
    for i in range(len(data_0)):
        data_0[i] = data_0[i].strip().split(' ')
        for j in range(9):
            data_0[i][j] = int(data_0[i][j])
        data_0[i][9] = round(float(data_0[i][9]), 4)
        data_0[i] = sort_transfer(data_0[i])
    data_1 = data1.strip().split('\n')
    for i in range(len(data_1)):
        data_1[i] = data_1[i].strip().split(' ')
        for j in range(9):
            data_1[i][j] = int(data_1[i][j])
        data_1[i][9] = round(float(data_1[i][9]), 4)
        data_1[i] = sort_transfer(data_1[i])
    data_tof = datatof.strip().split('\n')
    for i in range(len(data_tof)):
        data_tof[i] = data_tof[i].strip().split(' ')
        for j in range(500):
            data_tof[i][j] = int(data_tof[i][j])
        data_tof[i][500] = round(float(data_tof[i][500]), 4)

    linenum_0 = 0
    linenum_1 = 0
    linenum_tof = 0
    start_time = timeit.default_timer()
    while 1:
        current_time = timeit.default_timer() - start_time
        if linenum_0 <= len(data_0)-1 and current_time >= data_0[linenum_0][9]:
            for i in range(3):
                for j in range(3):
                    heat_map0[i][j] = data_0[linenum_0][i*3+j]
            linenum_0 += 1
        if linenum_1 <= len(data_1)-1 and current_time >= data_1[linenum_1][9]:
            # fresh the heat map of sensor 1
            for i in range(3):
                for j in range(3):
                    heat_map1[i][j] = data_1[linenum_1][i*3+j]
            linenum_1 += 1
        if linenum_tof <= len(data_tof)-1 and current_time >= data_tof[linenum_tof][500]:
            # fresh the heat map of tof
            # read the line to 25x20
            for i in range(25):
                for j in range(20):
                    tof_map[i][j] = data_tof[linenum_tof][i*20+j]

            # transfer map from 25x20 to 20x25
            for i in range(25):
                for j in range(20):
                    heat_map_tof[j][i] = tof_map[i][j]
            linenum_tof += 1

        if linenum_0 > len(data_0)-1 and linenum_1 > len(data_1)-1 and linenum_tof > len(data_tof)-1:
            break
        heatmap()
        time.sleep(0.02)
        root.update()
    print(int(timeit.default_timer() - start_time))
    print("Done")
    # root.protocol('WM_DELETE_WINDOW', OnExit)


def OnExit():
    root.destroy()


# subject = input("subject:")
# ac_num = input("activity number:")
filename0 = input("filename0:")  # subject + '_' + ac_num + '_0'
filename1 = input("filename1:")  # subject + '_' + ac_num + '_1'
filename_tof = input("filename_tof:")  # subject+'_'+ ac_num + '_tof'
f0 = open(filename0 + '.txt', 'r')
f1 = open(filename1 + '.txt', 'r')
f2 = open(filename_tof + '.txt', 'r')
heat_min = 0
heat_max = 4500
heat_map0 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
heat_map1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
tof_map = [[0 for j in range(20)] for i in range(25)]  # first read as 25x20, then transfer to 20x25
heat_map_tof = [[0 for j in range(25)] for i in range(20)]
palette = (1, 0, 0), (1, 1, 0), (0, 1, 0)
root = Tk()
root.title('Heatmap')
frame = Frame(root, width=900, height=900)
frame.pack()
# img = PhotoImage(file="bed.ppm")
panel = Label(frame)  # , image = img)

width, height = 300, 300  # Canvas size.
rows, cols = 3, 3
rect_width, rect_height = width // rows, height // cols
border = 0
width_tof, height_tof = 500,400
canvas1 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas2 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas3 = Canvas(frame, width=width_tof, height=height_tof, borderwidth=0, highlightthickness=0)
canvas1.place(relx=0.1, rely=0.5)
canvas2.place(relx=0.6, rely=0.5)
canvas3.place(relx=0.3, rely=0.02)
panel.place(relx=0, rely=0)
root.resizable(width=False, height=False)
d0 = f0.read()
d1 = f1.read()
d_tof = f2.read()
main(d0, d1, d_tof)
