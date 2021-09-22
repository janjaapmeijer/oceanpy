import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def cmap_sea_surface():
    c = mcolors.ColorConverter().to_rgb
    seq = [c('navy'), c('lightseagreen'), 0.45,
     c('lightseagreen'), c('lightsage'), 0.75,
     c('lightsage'), c('lightyellow'), 0.99, c('white')]
    sea_surface = make_colormap(seq)
    return sea_surface

def cmap_sea_surface_r():
    c = mcolors.ColorConverter().to_rgb
    seq = [c('white'), 0.01, c('lightyellow'),
           c('lightsage'), 0.25, c('lightsage'),
           c('lightseagreen'), 0.55, c('lightseagreen'), c('navy')]
    sea_surface = make_colormap(seq)
    return sea_surface


def set_color_cmap(cmap, ncolors, nset, color):
    cmapcolors = cmap(np.linspace(0, 1, ncolors))
    cmaplist = [tuple(cmapcolor) for cmapcolor in cmapcolors]
    cmaplist[nset] = mcolors.colorConverter.to_rgba(color)
    cmap = cmap.from_list('cmap', cmaplist, cmap.N)
    return cmap

# def extend_colorbar(cmap, color, extreme='min'):
#     cmap.set_under(color)
#     plt.colorbar(extend=extreme)
#     return cmap


# c = mcolors.ColorConverter().to_rgb
# # rvb = make_colormap(
# #     [c('red'), c('violet'), 0.33, c('violet'), c('blue'), 0.66, c('blue')])
# rvb = make_colormap(
#     [c('navy'), c('lightseagreen'), 0.45,
#      c('lightseagreen'), c('lightsage'), 0.75,
#      c('lightsage'), c('lightyellow'), 0.99, c('white')])
# N = 1000
# array_dg = np.random.uniform(0, 10, size=(N, 2))
# colors = np.random.uniform(-2, 2, size=(N,))
# plt.figure()
# plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=rvb)
# plt.colorbar()
# plt.show()