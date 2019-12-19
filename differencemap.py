"""
single pod difference map.
The processing is simple: blank(background)_data - current_data
then visualize it.

Since different activities' results from the subtraction is very different,
you should edit the maxval in colorize function to statisfy current conditions.
"""
try:
    from Tkinter import *
except ImportError:
    from tkinter import *  # Python 3
import numpy as np
import time
global heat_map


def sub_array(l1, l2):
    temp = []
    for i in range(9):
        temp.append(l1[i] - l2[i])
    return temp


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
    canvas1.delete('all')
    # canvas2.delete('all')
    # canvas3.delete('all')

    heat_map1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            heat_map1[i][j] = max(heat_map[i * 3 + j], 0)

    for y, row in enumerate(heat_map1):
        for x, temp in enumerate(row):
            x0, y0 = x * rect_width, y * rect_height
            x1, y1 = x0 + rect_width - border, y0 + rect_height - border
            color = colorize(temp, 0, 100, palette)
            canvas1.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)



def OnExit():
    root.destroy()


def check(list1, list2):
    for i in range(9):
        if list1[i] != list2[i]:
            return False
    return True


f1 = input("blank filename:")
f2 = input("data filename:")

# while import data, ignore the repeated ones
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
length = 0
temp_prev = [0, 0, 0, 0, 0, 0, 0, 0, 0]
temp_current = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(r1)):
    temp_current = r1[i]
    if check(temp_current, temp_prev):
        continue
    temp_prev = r1[i]
    for j in range(9):
        sum_blank[j] += r1[i][j]
    length += 1
avg_blank = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(9):
    avg_blank[i] = int(sum_blank[i] / length)

data_file = open(f2 + ".txt", 'r').read()
data_file = data_file.strip()
data = data_file.split('\n')
r2 = []
for line in data:
    line = line.strip()
    l = line.split(' ')
    temp = []
    for i in range(9):
        temp.append(int(l[i]))
    r2.append(temp.copy())
sum_data = [0, 0, 0, 0, 0, 0, 0, 0, 0]
temp_prev = [0, 0, 0, 0, 0, 0, 0, 0, 0]
temp_current = [0, 0, 0, 0, 0, 0, 0, 0, 0]
length2 = 0
for i in range(len(r2)):
    temp_current = r2[i]
    if check(temp_current, temp_prev):
        continue
    temp_prev = r2[i]
    length2 += 1
    for j in range(9):
        sum_data[j] += r2[i][j]
avg_data = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(9):
    avg_data[i] = int(sum_data[i] / length2)

heat_map = sub_array(avg_blank, avg_data)
palette = (1, 0, 0), (1, 1, 0), (0, 1, 0)
root = Tk()
root.title('difference map')
frame = Frame(root, width=600, height=600)
frame.pack()
# img = PhotoImage(file="bed.ppm")
panel = Label(frame)  # , image = img)

width, height = 300, 300  # Canvas size.
rows, cols = 3, 3
rect_width, rect_height = width // rows, height // cols
border = 0
canvas1 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas1.place(relx=0.25, rely=0.25)
panel.place(relx=0, rely=0)
root.resizable(width=False, height=False)
print(heat_map)
heatmap()
root.mainloop()
