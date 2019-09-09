try:
    from Tkinter import *
except ImportError:
    from tkinter import *  # Python 3
import numpy as np
import time

global heat_map


########################################################################
# This program loads in the distance value stored in distances.npy file,
# and update the graphic display according to the distance. 
########################################################################


########################################################################
# This function assigns color to the number range
########################################################################
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


########################################################################
# This function clears the canvases and draw the heap map accoridng to
# the updated data
########################################################################
def heatmap():
    canvas1.delete('all')
    canvas2.delete('all')
    canvas3.delete('all')
    try:
        heat_map = np.load('distances.npy')
    except:
        return None
    if (len(heat_map) != 3): return None

    heat_map1 = heat_map[0]
    print(heat_map)
    heat_map2 = heat_map[1]
    heat_map3 = heat_map[2]
    # print(heat_map1)
    # draw the heatmap
    for y, row in enumerate(heat_map1):
        for x, temp in enumerate(row):
            x0, y0 = x * rect_width, y * rect_height
            x1, y1 = x0 + rect_width - border, y0 + rect_height - border
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
            x0, y0 = x * rect_width, y * rect_height
            x1, y1 = x0 + rect_width - border, y0 + rect_height - border
            color = colorize(temp, heat_min, heat_max, palette)
            canvas3.create_rectangle(x0, y0, x1, y1, fill=color, width=0, stipple=None)


def rand():  # for testing
    heat_map_rnd = np.random.rand(3, 3, 3)
    print(heat_map_rnd)
    np.save('distances', heat_map_rnd)


def main():
    while 1:
        heatmap()
        time.sleep(0.2)
        root.update()
        # root.protocol('WM_DELETE_WINDOW', OnExit)
        print("test!!!")


def OnExit():
    root.destroy()


# max and min value of the heatmap
heat_min = 0
heat_max = 3500

# Heatmap rgb colors in mapping order (ascending).
# We want green->yello->red, no blues
heat_map = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]

palette = (1, 0, 0), (1, 1, 0), (0, 1, 0)
root = Tk()
root.title('Heatmap')
# Create frame
frame = Frame(root, width=1200, height=600)  #
frame.pack()
# load background image
img = PhotoImage(file="all.ppm")
panel = Label(frame, image=img)

# Calculate the size of each cell
width, height = 150, 150  # Canvas size.
rows, cols = 3, 3
rect_width, rect_height = width // rows, height // cols
border = 0  # Pixel width of border around each.
# Create canvases
canvas1 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas2 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)
canvas3 = Canvas(frame, width=width, height=height, borderwidth=0, highlightthickness=0)

# Place canvises, coordinate is relative
canvas1.place(relx=0.845, rely=0.04)
canvas2.place(relx=0.845, rely=0.7)
canvas3.place(relx=0.05, rely=0.35)
panel.place(relx=0, rely=0)
# paneld.place(relx = 0.85, rely=0)

root.resizable(width=False, height=False)
main()
