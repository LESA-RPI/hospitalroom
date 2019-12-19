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
import json
import pickle
from PIL import Image, ImageTk
from block_class import *

global heat_map
global weights
global bias
global block_frames


def pseudocolor(value, minval, maxval, palette):
    """ Maps given value to a linearly interpolated palette color. """
    max_index = len(palette)-1
    # Convert value in range minval...maxval to the range 0..max_index.
    v = (float(min(max(value,minval),maxval)-minval) / (maxval-minval)) * max_index
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

def sort_map2(orig_map):
    # the sort way is different from visualizing way
    # this is the original version, real time visualization requires corresponding with observation angle
    temp_list=[0,0,0,0,0,0,0,0,0]
    temp_list[5] = orig_map[0]
    temp_list[8] = orig_map[1]
    temp_list[7] = orig_map[2]
    temp_list[2] = orig_map[3]
    temp_list[1] = orig_map[4]
    temp_list[4] = orig_map[5]
    temp_list[0] = orig_map[6]
    temp_list[6] = orig_map[7]
    temp_list[3] = orig_map[8]
    return temp_list


def action_num(data0,data1):
    data1x18 = data0 + data1
    layer = np.matmul(data1x18,weights)+ bias
    label_estimate = np.argmax(layer)
    return label_estimate


def heatmap():
    # represent Webcam in Tkinter window
    # since it is more stable and has higher frequency, deal it at first
    canvas_cam.delete('all')
    ret, frame_video = cap.read()
    temp_frame = frame_video  # if the image needs to flip: cv2.flip(frame_video, 1)
    cv2image = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img = img.resize(( int(640*0.7), int(480*0.7) ), Image.ANTIALIAS)  # resize the image
    canvas_cam.image = ImageTk.PhotoImage(img)
    canvas_cam.create_image(0,0,anchor='nw',image=canvas_cam.image)

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
    heat_map1_orig = heat_map['arpae/192.168.0.101']['distances']  # change path here if have different folder design
    heat_map2_orig = heat_map['arpae/192.168.0.102']['distances']  # change path here if have different folder design
    heat_map1 = sort_map(heat_map1_orig)
    heat_map2 = sort_map(heat_map2_orig)
    data1x9_sensor0 = sort_map2(heat_map1_orig)
    data1x9_sensor1 = sort_map2(heat_map2_orig)
    action_id = action_num(data1x9_sensor0,data1x9_sensor1)
    current_map_18 = data1x9_sensor0 + data1x9_sensor1
    if block_frames.get_length()[0] < 8:
        block_frames.add_frame(current_map_18.copy(),action_id)
    else:
        if block_frames.check_frame_different(current_map_18):
            block_frames.add_frame(current_map_18.copy(),action_id)

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
            x0, y0 = x * 12, y * 12
            x1, y1 = x0 + 12 - border, y0 + 12 - border
            color = colorize(temp, heat_min, heat_max, palette)
            canvas3.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)
    print(block_frames.decision())

def main():

    while 1:


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

#%% main function starts here


# initialize the Webcam, 0 or 1 or another integer depends on USB connection
cap = cv2.VideoCapture(0)  # change the 0 if u have multiple cam connections

# set the range of the 9 pixel sensor' data
heat_min = 500
heat_max = 3000  # to be tested. 6000 is fine, but should be as small as possible
heat_map = [ [[0,0,0],[0,0,0],[0,0,0]] ,[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]
palette = (1, 0, 0), (1, 1, 0), (0, 1, 0)
# create the heat map drawing
root = Tk()
root.title('hospital')
frame = Frame(root, width=1400, height=700)
frame.pack()
panel = Label(frame)

# determine the 9 pixel sensor's size
width, height = 270, 270  # Canvas size.
rows, cols = 3, 3
rect_width, rect_height = width // rows, height // cols
border = 0
width_3, height_3 = 270, 216  # the size of 25x20 tof cam
height_colorband = (heat_max-heat_min)/10

# set up the canvas
canvas1 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas2 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas3 = Canvas(frame, width=width_3, height=height_3, borderwidth=0, highlightthickness=0)

canvas1_bg = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas1_bg.pack(expand = YES, fill = BOTH)
canvas1_bg.image = ImageTk.PhotoImage(Image.open("map_background_0.jpg").resize(( width,height), Image.ANTIALIAS))
canvas1_bg.create_image(0,0,anchor='nw',image=canvas1_bg.image)

canvas2_bg = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas2_bg.pack(expand = YES, fill = BOTH)
canvas2_bg.image = ImageTk.PhotoImage(Image.open("map_background_1.jpg").resize((width,height),Image.ANTIALIAS))
canvas2_bg.create_image(0,0,anchor='nw',image=canvas2_bg.image)

canvas3_bg = Canvas(frame, width=width_3, height=280, borderwidth=0, highlightthickness=0)
canvas3_bg.pack(expand = YES, fill = BOTH)
canvas3_bg.image = ImageTk.PhotoImage(Image.open("map_background_tof.jpg").resize((width, height),Image.ANTIALIAS))
canvas3_bg.create_image(0,0,anchor='nw',image=canvas3_bg.image)

ret, frame_video = cap.read()
height_cam, width_cam = frame_video.shape[:2]
# print(height_cam,width_cam)
canvas_cam = Canvas(frame, width=width_cam*0.7, height=height_cam*0.7,borderwidth=0, highlightthickness=0)
canvas_cam.pack(expand = YES, fill = BOTH)

# there is a color band assistance visualization recognition.
canvas4 = Canvas(frame, width=20, height=height_colorband, borderwidth=0, highlightthickness=0)

# place the three canvas
"""the place position and the size needs to be changed"""
canvas1.place(relx=0.05, rely=0.02)
canvas2.place(relx=0.25, rely=0.02)
canvas3.place(relx=0.45, rely=0.05)
canvas4.place(relx=0.01, rely=0.01)
canvas1_bg.place(relx=0.05, rely=0.5)
canvas2_bg.place(relx=0.25, rely=0.5)
canvas3_bg.place(relx=0.45, rely=0.5)
canvas_cam.place(relx=0.65, rely=0.2)

panel.place(relx=0, rely=0)
root.resizable(width=False, height=False)
# start the infinite loop

# draw the color band which will not be refresh
for i in range(0, int((heat_max-heat_min)/50)):
    x0, x1 = 0, 20
    y0 = i*5
    y1 = y0 + 5 - border
    color = colorize(heat_min+i*50, heat_min, heat_max, palette)
    canvas4.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)

# import the weight matrix from machine learning
""" 
the weights is an array that includes two parts: weights & bias
weights: 18x[1x7] input notes
bias:[1x7]
"""
f = open('weights_5act.txt', 'rb')  # HL_hospital_tf_weight_saving/
w = pickle.load(f)
weights = list(w[0][0])
bias = list(w[1][0])
# create the Frames class to contain the current block
block_frames = Frames(weights, bias)

# start infinite loop function
main()
