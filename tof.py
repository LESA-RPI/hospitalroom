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
        for i in range(25):
            for j in range(20):
                heat_map[i][j] = int(heat_map[i][j])
    except:
        return None

    # print("heat map:\n", heat_map)
    for y, row in enumerate(heat_map):
        for x, temp in enumerate(row):
            x0, y0 = x * rect_width, y * rect_height
            x1, y1 = x0 + rect_width - border, y0 + rect_height - border
            color = colorize(temp, heat_min, heat_max, palette)
            canvas1.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)


def main():
    while 1:
        heatmap()
        time.sleep(0.1)
        root.update()


def OnExit():
    root.destroy()


heat_min = 0
heat_max = 5000
heat_map = [[0 for j in range(20)] for i in range(25)]
# print(len(heat_map), len(heat_map[1]))
palette = (1, 0, 0), (1, 1, 0), (0, 1, 0)
root = Tk()
root.title('TOF')
frame = Frame(root, width=800, height=600)
frame.pack()
panel = Label(frame)  # , image = img)

width, height = 500, 400  # Canvas size.
rows, cols = 25, 20
rect_width, rect_height = 20, 20
border = 0
canvas1 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas1.place(relx=0.1, rely=0.1)
panel.place(relx=0, rely=0)
root.resizable(width=False, height=False)
main()
