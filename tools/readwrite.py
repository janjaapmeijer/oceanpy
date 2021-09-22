import numpy as np
import os

def readxyz(filename,step=None,xy=False):

    xyz = open(filename)
    if xy is False:
        x = []
        y = []
        z = []

        for line in xyz:
            x1,y1,z1 = line.split()
            x.append(float(x1))
            y.append(float(y1))
            if z1 == '-9999':
                z.append(float('nan'))
            else:
                z.append(float(z1))
        xyz.close()

        if step is not None:
            xstep  = [x[i] for i in range(0,len(x),step)]
            ystep  = [y[i] for i in range(0,len(y),step)]
            zstep  = [z[i] for i in range(0,len(z),step)]

            return xstep, ystep, zstep

        else:
            return x, y, z

    else:
        x = []
        y = []

        for line in xyz:
            x1,y1 = line.split()
            x.append(float(x1))
            y.append(float(y1))
        xyz.close()

        return (x, y)

def writexyz(filename, x, y, z=None):

    if z is None:
        with open(filename, "w") as xy:

            for i in range(0,len(x)):
                xy.write(str(x[i]) + " " + str(y[i]) + '\n')

    else:
        with open(filename, "w") as xyz:
            if isinstance(x, list):
                for i in range(0,len(x)):
                    xyz.write(str(x[i]) + " " + str(y[i]) + " " + str(z[i]) + '\n')
            else:
                xyz.write(str(x) + " " + str(y) + " " + str(z))

def addxyz(filenamelst,step=None):

    x = []
    y = []
    z = []

    for i in range(0,len(filenamelst)):

        xtemp, ytemp, ztemp = readxyz(filenamelst[i],step)

        x = x + xtemp
        y = y + ytemp
        z = z + ztemp

        del xtemp, ytemp, ztemp

    return x, y, z

def readxytxt(filename):
    x = np.genfromtxt(filename, usecols=(0), delimiter=' ', dtype=None) # comma: delimiter=','
    y = np.genfromtxt(filename, usecols=(1), delimiter=' ', dtype=None)

    return x, y

def write_ascii(filename, array, xll, yll, cellsize, nodata=int(-9999)):
    import collections

    # header = {'ncols': array.shape[1], 'nrows': array.shape[0], 'xllcorner': xll, 'yllcorner': yll,
    #           'cellsize': cellsize, 'NODATA_value': nodata}

    header = collections.OrderedDict((('ncols', array.shape[1]),
                                     ('nrows', array.shape[0]),
                                     ('xllcorner', xll),
                                     ('yllcorner', yll),
                                     ('cellsize', cellsize),
                                     ('NODATA_value', nodata)))

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    array[np.isnan(array)] = nodata
    # np.savetxt(filename, np.flipud(array), header=header, comments='', fmt="%1.4f")

    delimiter = '\t'
    fmt="%1.4f"
    with open(filename, 'w') as f:
        for key in header.keys():
            f.write(key + '\t%s\n' % header[key])
        for row in np.flipud(array):
            line = delimiter.join("-9999" if value == nodata else fmt % value for value in row)
            f.write(line + '\n')


def read_asciiheader(filename, info=None):

    import linecache
    if info is None:
        info = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'NODATA_value']

    header = {}
    for (i, var) in enumerate(info):
        var, val = linecache.getline(filename, i+1).split()
        header[var] = float(val)
    return header


def read_ascii(filename, nodata='-9999'):

    header = read_asciiheader(filename)

    array = np.flipud(np.genfromtxt(filename, skip_header=6, missing_values=nodata, usemask=True).filled(np.nan))

    return array, header


def readxls(filename, colnos, rowstart = 0, sheetno = 0):
    import xlrd

    workbook = xlrd.open_workbook(filename)
    sheet = workbook.sheet_by_index(sheetno)

    x = []
    y = []

    for row in range(rowstart,sheet.nrows):
        x.append(sheet.cell_value(row,colnos[0]))
        y.append(sheet.cell_value(row,colnos[1]))
    return x, y

import pickle

def write_dict(dictionary, path, filename, protocol=pickle.HIGHEST_PROTOCOL):
    with open(os.path.join(path, filename + '.pkl'), 'wb') as f:
        pickle.dump(dictionary, f, protocol=protocol)


def read_dict(path, filename, encoding='ascii'):
    with open(os.path.join(path, filename), 'rb') as f:
        return pickle.load(f, encoding=encoding)
