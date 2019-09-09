try:
    from Tkinter import *
except ImportError:
    from tkinter import *  # Python 3
import time

global heat_map


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


def heatmap():
    canvas1.delete('all')
    try:
        data_file = open('scr.tof_control/SCR/output-0.txt', 'r')
        data = data_file.read().strip()
        heat_map = data.split('\n')
        for i in range(25):
            heat_map[i] = heat_map[i].strip().split('\t')
        tof_map = [[0 for j in range(25)] for i in range(20)]
        for i in range(25):
            for j in range(20):
                tof_map[j][i] = int(heat_map[i][j])
    except:
        return None
    # print(len(tof_map), len(tof_map[0]))
    # print("heat map:\n", heat_map)
    tof_map_10x10 = transfer10x10(tof_map)
    tof_map_8x8 = transfer8x8(tof_map)
    tof_map_4x4 = transfer_4x4(tof_map)
    for y, row in enumerate(tof_map_10x10):
        for x, temp in enumerate(row):
            x0, y0 = x * rect_width, y * rect_height
            x1, y1 = x0 + rect_width - border, y0 + rect_height - border
            color = colorize(temp, heat_min, heat_max, palette)
            canvas1.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)

    for y, row in enumerate(tof_map_8x8):
        for x, temp in enumerate(row):
            x0, y0 = x * rect_width, y * rect_height
            x1, y1 = x0 + rect_width - border, y0 + rect_height - border
            color = colorize(temp, heat_min, heat_max, palette)
            canvas2.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)

    for y, row in enumerate(tof_map_4x4):
        for x, temp in enumerate(row):
            x0, y0 = x * rect_width, y * rect_height
            x1, y1 = x0 + rect_width - border, y0 + rect_height - border
            color = colorize(temp, heat_min, heat_max, palette)
            canvas3.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)

    for y, row in enumerate(tof_map):
        for x, temp in enumerate(row):
            x0, y0 = x * 20, y * 20
            x1, y1 = x0 + 20 - border, y0 + 20 - border
            color = colorize(temp, heat_min, heat_max, palette)
            canvas4.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)


def main():
    while 1:
        heatmap()
        time.sleep(0.05)
        root.update()


def OnExit():
    root.destroy()


def avg_of_area(data_map, n_row, n_col, start_i, start_j):
    sum_of_area = 0
    for i in range(n_row):
        for j in range(n_col):
            sum_of_area += data_map[start_i+i][start_j+j]
    avg = int(sum_of_area/(n_row*n_col))
    return avg


# 20x25 -> 8x8, 10x10
def transfer8x8(tof_map):
    temp_map = [[0 for j in range(8)] for i in range(8)]
    i, j = 0, 0
    x, y = 0, 0
    while i < 20 and x < 10:
        if j == 21:
            temp_map[x][y] = avg_of_area(tof_map, 3, 4, i, j)
        else:
            temp_map[x][y] = avg_of_area(tof_map, 3, 3, i, j)
        j += 3
        if j == 24:
            j = 0
            y = 0
            x += 1
            if i % 5 == 0:
                i += 2
            elif i % 5 == 2:
                i += 3
            else:
                print("row value wrong")
        else:
            y += 1

    return temp_map


def transfer10x10(tof_map):
    temp_map = [[0 for j in range(10)] for i in range(10)]
    i, j = 0, 0
    x, y = 0, 0
    while i < 20 and x < 10:
        temp_map[x][y] = avg_of_area(tof_map, 2, 3, i, j)
        if j % 5 == 0:
            j += 2
            y += 1
        elif j % 5 == 2 and j != 22:
            j += 3
            y += 1
        elif j == 22:
            x += 1
            i += 2
            y = 0
            j = 0
        else:
            print("col value wrong")
        # print("(i,j):",i,j,"(x,y)",x,y)
    return temp_map


def transfer_4x4(tof_map):
    temp_map = [[0 for j in range(4)] for i in range(4)]
    i, j = 0, 0
    x, y = 0, 0
    while i < 20 and x < 4:
        if j == 18:
            temp_map[x][y] = avg_of_area(tof_map, 5, 7, i, j)
            y = 0
            j = 0
            x += 1
            i += 5
        else:
            temp_map[x][y] = avg_of_area(tof_map, 5, 6, i, j)
            y += 1
            j += 6
    return temp_map


heat_min = 0
heat_max = 5000
heat_map = [[0 for j in range(20)] for i in range(25)]
# print(len(heat_map), len(heat_map[1]))
palette = (1, 0, 0), (1, 1, 0), (0, 1, 0)
root = Tk()
root.title('TOF')
frame = Frame(root, width=1500, height=500)
frame.pack()
panel = Label(frame)  # , image = img)

width, height = 500, 400  # Canvas size.
rows, cols = 25, 20
rect_width, rect_height = 40, 40
border = 0
canvas1 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas2 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas3 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas4 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas1.place(relx=0.36, rely=0.1)
canvas2.place(relx=0.64, rely=0.1)
canvas3.place(relx=0.88, rely=0.1)
canvas4.place(relx=0.01, rely=0.1)
panel.place(relx=0, rely=0)
root.resizable(width=False, height=False)
main()
