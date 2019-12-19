"""
like height_transfer.py, but this one is real time visualization.
This program is used for transferring the data to direct distance from floor.
This processing is very rough and cannot be used for practical use. More like my personal temporary stuff.
"""
try:
    from Tkinter import *
except ImportError:
    from tkinter import *  # Python 3
import numpy as np
import cv2
import time
import json

global heat_map
global P2F  # the height of pod to the floor
global P2F_reading_1  # the reading value from the sensor
global P2F_reading_0
global ratio_0
global ratio_1
global base_line_1
global base_line_0
global base_distance_0
global text1
global text0


# %%
# transfer functions
def num_to_height(num, sensor_num):
    if sensor_num == 0:
        x = (P2F / P2F_reading_0) * num
    else:
        x = (P2F / P2F_reading_1) * num
    return round(x, 2)


# cos(30)=0.866, output a 3x3 matrix
def transfer_30_degree(heat_map):
    temp = [[0 for j in range(3)] for i in range(3)]
    difference = [[(base_line_0[i][j] - int(heat_map[i][j])) for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            temp[i][j] = round(num_to_height(int(difference[i][j] * ratio_0[i][j]), 0) + base_distance_0[i][j], 2)
    return temp


# cos(15)=0.966, output a 3x3 matrix
def transfer_15_degree(heat_map):
    temp = [[0 for j in range(3)] for i in range(3)]
    difference = [[(base_line_1[i][j] - int(heat_map[i][j])) for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            temp[i][j] = round(num_to_height(int(difference[i][j] * ratio_1[i][j]), 1) + base_distance_1[i][j], 2)
    return temp


# transfer the data_save format(1x9) to matrix format with correct sequence
def format_transfer(orig_1x9):
    temp_3x3 = [[0 for j in range(3)] for i in range(3)]
    temp_3x3[0][0] = orig_1x9[2]
    temp_3x3[0][1] = orig_1x9[5]
    temp_3x3[0][2] = orig_1x9[8]
    temp_3x3[1][0] = orig_1x9[1]
    temp_3x3[1][1] = orig_1x9[4]
    temp_3x3[1][2] = orig_1x9[7]
    temp_3x3[2][0] = orig_1x9[0]
    temp_3x3[2][1] = orig_1x9[3]
    temp_3x3[2][2] = orig_1x9[6]
    return temp_3x3


# %%
# heat map functions, copy and edit from hospitalmap.py
def pseudocolor(value, minval, maxval, palette):
    """ Maps given value to a linearly interpolated palette color. """
    max_index = len(palette) - 1
    # Convert value in range minval...maxval to the range 0..max_index.
    v = (float(value - minval) / (maxval - minval)) * max_index
    i = int(v);
    f = v - i  # Split into integer and fractional portions.
    c0r, c0g, c0b = palette[i]
    c1r, c1g, c1b = palette[min(i + 1, max_index)]
    dr, dg, db = c1r - c0r, c1g - c0g, c1b - c0b
    return c0r + (f * dr), c0g + (f * dg), c0b + (f * db)  # Linear interpolation.


def colorize(value, minval, maxval, palette):
    """ Convert value to heatmap color and convert it to tkinter color. """
    color = (int(c * 255) for c in pseudocolor(value, minval, maxval, palette))
    return '#{:02x}{:02x}{:02x}'.format(*color)  # Convert to hex string.


def sort_map(orig_map):
    # the sort way is different from the saving way
    # sort the map in right sequence. Correspond to the observer view
    temp_list = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    temp_list[0][1] = orig_map[0]
    temp_list[0][2] = orig_map[1]
    temp_list[1][2] = orig_map[2]
    temp_list[0][0] = orig_map[3]
    temp_list[1][0] = orig_map[4]
    temp_list[1][1] = orig_map[5]
    temp_list[2][0] = orig_map[6]
    temp_list[2][2] = orig_map[7]
    temp_list[2][1] = orig_map[8]
    return temp_list


def heatmap():
    try:
        # try open the three files.
        # If success, refresh the matrix data; if not, return None and go to next loop
        with open('ArpaE.9pixel/data/all_data.json') as json_file:
            heat_map = json.load(json_file)
            # print(heat_map)
    except:
        return None
    # clean the past canvas
    canvas1.delete('all')
    canvas2.delete('all')
    num_canvas0.delete('all')
    num_canvas1.delete('all')

    heat_map1_orig = heat_map['arpae/192.168.0.101']['distances']  # 0-> mid one, 1 flatten one
    heat_map2_orig = heat_map['arpae/192.168.0.102']['distances']
    heat_map1 = sort_map(heat_map1_orig)
    heat_map2 = sort_map(heat_map2_orig)
    height_map1 = transfer_30_degree(heat_map1)
    height_map2 = transfer_15_degree(heat_map2)

    # print("heat map1:\n", heat_map1)
    for y, row in enumerate(heat_map1):
        for x, temp in enumerate(row):
            x0, y0 = x * rect_width, y * rect_height
            x1, y1 = x0 + rect_width - border, y0 + rect_height - border
            color = colorize(temp, heat_min, heat_max, palette)
            canvas1.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)

    # print("heat map2:\n", heat_map2)
    for y, row in enumerate(heat_map2):
        for x, temp in enumerate(row):
            x0, y0 = x * rect_width, y * rect_height
            x1, y1 = x0 + rect_width - border, y0 + rect_height - border
            color = colorize(temp, heat_min, heat_max, palette)
            canvas2.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)

    text0.delete("1.0", "end")
    text1.delete("1.0", "end")
    for i in range(3):
        text0.insert(END, str(height_map1[i])+'\n')
        text1.insert(END, str(height_map2[i])+'\n')


def main():
    while 1:
        # Webcam capture
        ret, frame_video = cap.read()
        cv2.imshow('frame', frame_video)
        heatmap()
        time.sleep(0.1)
        root.update()  # root refresh
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # after the infinite loop, close the Webcam
    cap.release()
    cv2.destroyAllWindows()


def OnExit():
    root.destroy()


# %%
# preparation for height transfer
# this is based on the reading sequence, needs to be changed after all done
base_distance_0 = [0, 0, 0, 0.77, 0.77, 0.77, 0, 0, 0]
base_distance_1 = [0, 0, 0, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77]
P2F = 2.4
background_filename_0 = input("background filename sensor 0:")
f_blank_0 = open(background_filename_0 + '.txt', 'r')
data_blank_0 = f_blank_0.read()
data_blank_0 = data_blank_0.strip().split('\n')
for i in range(len(data_blank_0)):
    data_blank_0[i] = data_blank_0[i].strip().split(' ')
    for j in range(9):
        data_blank_0[i][j] = int(data_blank_0[i][j])
    data_blank_0[i][9] = float(data_blank_0[i][9])
sum_background_0 = [0 for i in range(9)]
for i in range(len(data_blank_0)):
    for j in range(9):
        sum_background_0[j] += data_blank_0[i][j]
base_line_0 = [int(sum_background_0[i] / len(data_blank_0)) for i in range(9)]
P2F_reading_0 = base_line_0[4] + 775

background_filename_1 = input("background filename sensor 1:")
f_blank_1 = open(background_filename_1 + ".txt", 'r')
data_blank_1 = f_blank_1.read()
data_blank_1 = data_blank_1.strip().split('\n')
for i in range(len(data_blank_1)):
    data_blank_1[i] = data_blank_1[i].strip().split(' ')
    for j in range(9):
        data_blank_1[i][j] = int(data_blank_1[i][j])
    data_blank_1[i][9] = float(data_blank_1[i][9])
sum_background_1 = [0 for i in range(9)]
for i in range(len(data_blank_1)):
    for j in range(9):
        sum_background_1[j] += data_blank_1[i][j]
base_line_1 = [int(sum_background_1[i] / len(data_blank_1)) for i in range(9)]
P2F_reading_1 = base_line_1[4] + 775

ratio_0 = [0 for i in range(9)]
ratio_1 = [0 for i in range(9)]
for i in range(9):
    if i < 3:
        ratio_1[i] = round(P2F_reading_1 / base_line_1[i], 2)
    else:
        ratio_1[i] = round(base_line_1[4] / base_line_1[i], 2)

for i in range(9):
    if 3 <= i <= 5:
        ratio_0[i] = round(base_line_0[4] / base_line_0[i], 2)
    else:
        ratio_0[i] = round(P2F_reading_0 / base_line_0[i], 2)

print(base_line_0)
print(ratio_0)
print(base_line_1)
print(ratio_1)
# change the base line and ratio for the visualizing sorting format. 1x9->3x3
base_line_0 = format_transfer(base_line_0)
ratio_0 = format_transfer(ratio_0)
base_line_1 = format_transfer(base_line_1)
ratio_1 = format_transfer(ratio_1)
base_distance_0 = format_transfer(base_distance_0)
base_distance_1 = format_transfer(base_distance_1)


# %%
# initialization for camera and heat map
cap = cv2.VideoCapture(0)  # initialize the Webcam, 0 or 1 or another integer depends on USB connection
# set the range of the 9 pixel sensor' data
heat_min = 500
heat_max = 4500  # to be tested. 6000 is fine, but should be as small as possible
heat_map = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
palette = (1, 0, 0), (1, 1, 0), (0, 1, 0)
# create the heat map drawing
root = Tk()
root.title('hospital with height')
frame = Frame(root, width=900, height=900)
frame.pack()
panel = Label(frame)

# determine the 9 pixel sensor's size
width, height = 300, 300  # Canvas size.
rows, cols = 3, 3
rect_width, rect_height = width // rows, height // cols
border = 0
width_3, height_3 = 500, 400  # the size of 25x20 tof cam
height_colorband = (heat_max - heat_min) / 10
canvas1 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas2 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas4 = Canvas(frame, width=5, height=height_colorband, borderwidth=0, highlightthickness=0)
num_canvas0 = Canvas(frame, width=width, height=height_colorband, borderwidth=0, highlightthickness=0)
num_canvas1 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)

text0 = Text(num_canvas0, width=30, height=10)
text1 = Text(num_canvas1, width=30, height=10)
text0.grid(row=0, column=0)
text1.grid(row=0, column=0)
text0.insert(INSERT, '1111')
text1.insert(INSERT, '2222')
text0.pack()
text1.pack()

# place the three canvas
canvas1.place(relx=0.1, rely=0.1)
canvas2.place(relx=0.6, rely=0.1)
canvas4.place(relx=0.99, rely=0)
num_canvas0.place(relx=0.1, rely=0.5)
num_canvas1.place(relx=0.6, rely=0.5)
panel.place(relx=0, rely=0)
root.resizable(width=False, height=False)
# start the infinite loop
for i in range(0, int((heat_max - heat_min) / 50)):
    x0, x1 = 0, 5
    y0 = i * 5
    y1 = y0 + 5 - border
    color = colorize(heat_min + i * 50, heat_min, heat_max, palette)
    canvas4.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)

main()
