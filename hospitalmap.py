"""
Real time visualization, including 3 parts in 2 windows:
1. 2 9-pixel pods heat map
2. 20x25 matrix heat map
3. WebCam

Make sure that the gate way is set up and running correctly before running this program.
"""
try:
    from Tkinter import *
except ImportError:
    from tkinter import *  # Python 3
import numpy as np
import cv2
import time
global heat_map
import json


def pseudocolor(value, minval, maxval, palette):
    """ Maps given value to a linearly interpolated palette color. """
    max_index = len(palette)-1
    # Convert value in range minval...maxval to the range 0..max_index.
    v = (float(value-minval) / (maxval-minval)) * max_index
    i = int(v); f = v-i  # Split into integer and fractional portions.
    c0r, c0g, c0b = palette[i]
    c1r, c1g, c1b = palette[min(i+1, max_index)]
    dr, dg, db = c1r-c0r, c1g-c0g, c1b-c0b
    return c0r+(f*dr), c0g+(f*dg), c0b+(f*db)  # Linear interpolation.


def colorize(value, minval, maxval, palette):
    """ Convert value to heatmap color and convert it to tkinter color. """
    color = (int(c*255) for c in pseudocolor(value, minval, maxval, palette))
    return '#{:02x}{:02x}{:02x}'.format(*color)  # Convert to hex string.


def sort_map(orig_map):
    # sort the map in right consequence. Correspond to the observer view
    temp_list=[[0,0,0],[0,0,0],[0,0,0]]
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
    # try open the three files.
    # If success, refresh the matrix data; if not, return None and go to next loop
    try:
        # open the 9-pixel pods data, the format is json
        with open('ArpaE.9pixel/data/all_data.json') as json_file: # change the path here if you have different folder design
            heat_map = json.load(json_file)

        # open the 20x25 matrix tof sensor data
        data_file = open('scr.tof_control/SCR/output-0.txt', 'r') # change the path here if you have different folder design
        # read the tof cam data
        data = data_file.read().strip()
        tof_map = data.split('\n')
        for i in range(25):
            tof_map[i] = (tof_map[i].strip()).split('\t')
        # initialize tof map
        heat_map3 = [[0 for j in range(25)] for i in range(20)]
        # rotate the original map
        for i in range(25):
            for j in range(20):
                heat_map3[j][i] = int(tof_map[i][j])
    except:
        return None
    # clean the past canvas
    canvas1.delete('all')
    canvas2.delete('all')
    canvas3.delete('all')

    # there are two pods sensors
    # 1 -> mid one, 2 -> flatten one
    heat_map1_orig = heat_map['arpae/192.168.0.101']['distances'] # change the path here if you have different folder design
    heat_map2_orig = heat_map['arpae/192.168.0.102']['distances'] # change the path here if you have different folder design
    heat_map1 = sort_map(heat_map1_orig)
    heat_map2 = sort_map(heat_map2_orig)

    # Tkinter drawing
    for y, row in enumerate(heat_map1):
        for x, temp in enumerate(row):
            x0, y0 = x * rect_width, y * rect_height
            x1, y1 = x0 + rect_width-border, y0 + rect_height-border
            color = colorize(temp, heat_min, heat_max, palette)
            canvas1.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)

    for y, row in enumerate(heat_map2):
        for x, temp in enumerate(row):
            x0, y0 = x * rect_width, y * rect_height
            x1, y1 = x0 + rect_width - border, y0 + rect_height - border
            color = colorize(temp, heat_min, heat_max, palette)
            canvas2.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)

    for y, row in enumerate(heat_map3):
        for x, temp in enumerate(row):
            x0, y0 = x * 20, y * 20
            x1, y1 = x0 + 20 - border, y0 + 20 - border
            color = colorize(temp, heat_min, heat_max, palette)
            canvas3.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)


def main():

    while 1:
        # Webcam capture
        ret, frame_video = cap.read()
        cv2.imshow('frame', frame_video)

        # heat map refresh
        heatmap()
        time.sleep(0.1)
        root.update() # root refresh

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # after the infinite loop, close the Webcam
    cap.release()
    cv2.destroyAllWindows()


def OnExit():
    root.destroy()

# initialize the Webcam, 0 or 1 or another integer depends on USB connection
cap = cv2.VideoCapture(0)  # change the 0 if u have multiple cam connections

# set the range of the 9 pixel sensor' data
heat_min = 500
heat_max = 4500  # to be tested. 6000 is fine, but should be as small as possible
heat_map = [ [[0,0,0],[0,0,0],[0,0,0]] ,[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]
palette = (1, 0, 0), (1, 1, 0), (0, 1, 0)
# create the heat map drawing
root = Tk()
root.title('hospital')
frame = Frame(root, width=900, height=900)
frame.pack()
panel = Label(frame)

# determine the 9 pixel sensor's size
width, height = 300, 300  # Canvas size.
rows, cols = 3, 3
rect_width, rect_height = width // rows, height // cols
border = 0
width_3, height_3 = 500, 400  # the size of 25x20 tof cam
height_colorband = (heat_max-heat_min)/10
canvas1 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas2 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas3 = Canvas(frame, width=width_3, height=height_3, borderwidth=0, highlightthickness=0)

#there is a color band assistance visualization recognition.
canvas4 = Canvas(frame, width=5, height=height_colorband, borderwidth=0, highlightthickness=0)

# place the three canvas
canvas1.place(relx=0.1, rely=0.5)
canvas2.place(relx=0.6, rely=0.5)
canvas3.place(relx=0.3, rely=0.02)
canvas4.place(relx=0.99, rely=0)
panel.place(relx=0, rely=0)
root.resizable(width=False, height=False)
# start the infinite loop

# draw the color band which will not be refresh
for i in range(0, int((heat_max-heat_min)/50)):
    x0, x1 = 0, 5
    y0 = i*5
    y1 = y0 + 5 - border
    color = colorize(heat_min+i*50, heat_min, heat_max, palette)
    canvas4.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)

# start infinite loop function
main()
