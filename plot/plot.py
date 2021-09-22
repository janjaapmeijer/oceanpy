from pylab import *

# http://stackoverflow.com/questions/24535848/drawing-log-linear-plot-on-a-square-plot-area-in-matplotlib
def linlogplot(fig):

    ax = gca()
    ax.set_xscale("log")

    # now get the figure size in real coordinates:
    fig  = gcf()
    fwidth = fig.get_figwidth()
    fheight = fig.get_figheight()

    # get the axis size and position in relative coordinates
    # this gives a BBox object
    bb = ax.get_position()

    # calculate them into real world coordinates
    axwidth = fwidth * (bb.x1 - bb.x0)
    axheight = fheight * (bb.y1 - bb.y0)

    # if the axis is wider than tall, then it has to be narrowe
    if axwidth > axheight:
        # calculate the narrowing relative to the figure
        narrow_by = (axwidth - axheight) / fwidth
        # move bounding box edges inwards the same amount to give the correct width
        bb.x0 += narrow_by / 2
        bb.x1 -= narrow_by / 2
    # else if the axis is taller than wide, make it vertically smaller
    # works the same as above
    elif axheight > axwidth:
        shrink_by = (axheight - axwidth) / fheight
        bb.y0 += shrink_by / 2
        bb.y1 -= shrink_by / 2

    ax.set_position(bb)
    show()

def loglinplot(fig):

    ax = gca()
    ax.set_yscale("log")

    # now get the figure size in real coordinates:
    fig  = gcf()
    fwidth = fig.get_figwidth()
    fheight = fig.get_figheight()

    # get the axis size and position in relative coordinates
    # this gives a BBox object
    bb = ax.get_position()

    # calculate them into real world coordinates
    axwidth = fwidth * (bb.x1 - bb.x0)
    axheight = fheight * (bb.y1 - bb.y0)

    # if the axis is wider than tall, then it has to be narrowe
    if axwidth > axheight:
        # calculate the narrowing relative to the figure
        narrow_by = (axwidth - axheight) / fwidth
        # move bounding box edges inwards the same amount to give the correct width
        bb.x0 += narrow_by / 2
        bb.x1 -= narrow_by / 2
    # else if the axis is taller than wide, make it vertically smaller
    # works the same as above
    elif axheight > axwidth:
        shrink_by = (axheight - axwidth) / fheight
        bb.y0 += shrink_by / 2
        bb.y1 -= shrink_by / 2

    ax.set_position(bb)
    show()

# # plot circle around point
# circ1000 = plt.Circle(x,y,1000, color='blue', fill=False)
# fig.gca().add_artist(circ1000)
