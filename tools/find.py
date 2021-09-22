
import numpy as np

def find(lstlon, lstlat, llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat):
    result = []
    for index, (i,j) in enumerate(zip(lstlon,lstlat)):
        if i >= llcrnrlon and i <= urcrnrlon and j >=llcrnrlat and j <= urcrnrlat:
            result.append(index)
    return result

""" FIND FLOATS/INTEGERS IN LIST/SERIES"""
def find_finite(lst):
    import numpy as np
    result = []
    for index, i in enumerate(lst):
        if np.isfinite(i):
            result.append(index)
    return result

""" FIND NAN IN LIST/SERIES OF FLOATS/INTEGERS"""
def find_nan(lst):
    import numpy as np
    result = []
    for index, i in enumerate(lst):
        if np.isnan(i):
            result.append(index)
    return result

""" FIND NAN IN LIST/SERIES OF STRINGS"""
def find_nanstr(lst_str):
    import pandas as pd
    result = []
    for index, i in enumerate(lst):
        if pd.isnull(i):
            result.append(index)
    return result

def find_str(lst_str,string):
    """ FIND STRING IN LIST/SERIES OF STRINGS"""

    import pandas as pd
    if type(string) == str:
        result = []
        for index, i in enumerate(lst_str):
            if i == string:
                result.append(index)
                return result
    else:
        print('input value is not a string')

# def find_value():
#     if array.ndim > 1:
#         for row, i in enumerate(array):
#             try:
#                 col = i.index(value)
#             except ValueError:
#                 continue
#             return row, col
#         return -1
#     else:
#         raise ValueError("Array must be 2-dimensional")

def find_closest(array, value, array2=None, value2=None, return_value=False):
    """ FIND INDEX IN LIST CLOSEST TO PREDEFINED VALUE """

    import numpy as np
    if not array2 == None and value2:
        id = (np.sqrt((array-value)**2 + (array2-value2)**2)).argmin()
        if return_value:
            return id, array[id]
        return id
    else:
        id = (np.abs(array-value)).argmin()
        if return_value:
            return id, array[id]
        return id

def closestgridpnt(coords, xgrid, ygrid):
    """ FIND COORDINATE IN GRID CLOSEST TO PREDEFINED COORDINATES """

    from scipy import spatial

    # gridID = []
    griddata = list(zip(xgrid.ravel(), ygrid.ravel()))
    gridcoords = []
    for coord in coords:
        _, index = spatial.KDTree(griddata).query(coord)
        # gridpt = grid[spatial.KDTree(grid).query(pt)[1]] # or grid[index]
        # gridID.append(grid_inds[index])
        gridcoords.append(griddata[index])
    return gridcoords

# def findy(lst, lat1, lat2):
#     result = []
#     for index, i in enumerate(lst):
#         if i >= lat1 and i < lat2:
#             result.append(index)
#     return result

def find_in_circle(x, y, center, radius):
    '''
    Find x,y-coordinates that are within a circle distance (radius) from a center coordinate.
    :param x:
    :param y:
    :param center:
    :param radius:
    :return: indices of points in circle
    '''

    x0, y0 = center
    idx = np.where((x - x0) ** 2 + (y - y0) ** 2 <= radius ** 2)[0]
    return idx