import numpy as np
from OceanPy.projections import rotatexy
from scipy.interpolate import griddata
from matplotlib.path import Path


def xyz_in_rectangle(x, y, z, ll, lr, ur, ul):
    domain = Path([ll, lr, ur, ul, ll], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])

    condition = domain.contains_points(list(zip(x, y)))
    xrect = np.extract(condition, x)
    yrect = np.extract(condition, y)
    zrect = np.extract(condition, z)

    return xrect, yrect, zrect

def ascii(x, y, z, dx, dy):

    nx = int((max(x) - min(x)) / dx)+1
    ny = int((max(y) - min(y)) / dy)+1
    xi = np.linspace(min(x), max(x), nx)
    yi = np.linspace(min(y), max(y), ny)

    xgrid, ygrid = np.meshgrid(xi, yi)
    vargrid = griddata((x, y), z, (xgrid, ygrid), method='linear')

    return xgrid, ygrid, vargrid

def ascii_fromheader(header):
    xi = np.linspace(header['xllcorner'], header['xllcorner'] + header['cellsize'] * header['ncols'], header['ncols'],
                     endpoint=False)
    yi = np.linspace(header['yllcorner'], header['yllcorner'] + header['cellsize'] * header['nrows'], header['nrows'],
                     endpoint=False)
    xgrid, ygrid = np.meshgrid(xi, yi)

    return xgrid, ygrid


def xbeach(x, y, z, xori, yori, alfa, dist_cross, dist_along, dx, dy): #, dist_cross, dist_along
    ''' alfa is the coastal orientation angle '''

    rot_coords = rotatexy(xori, yori, x, y, alfa)
    xprime, yprime = list(zip(*rot_coords))

    nx = int(dist_cross / dx) + 1
    ny = int(dist_along / dy) + 1
    xi = np.linspace(xori, xori+dist_cross, nx)
    yi = np.linspace(yori, yori+dist_along, ny)

    xgrid, ygrid = np.meshgrid(xi, yi)
    zb = griddata((xprime, yprime), z, (xgrid, ygrid), method='linear')

    xgrid_flat = np.reshape(xgrid, np.prod([xgrid.shape]))
    ygrid_flat = np.reshape(ygrid, np.prod([ygrid.shape]))

    grid_coords = rotatexy(xori, yori, xgrid_flat, ygrid_flat, -alfa)

    xg, yg = list(zip(*grid_coords))

    xgrid = np.reshape(xg, xgrid.shape)
    ygrid = np.reshape(yg, ygrid.shape)

    return xgrid, ygrid, zb

# TODO: xgrd, ygrd, x0, y0 check 1dim or 2dim, if 2dim flatten
# TODO: cartesion or polar if cartesian np.sqrt((xgrd-x0)**2 + (ygrd-y0)**2), elif polar haversine
def closest_gridpnt(longrd, latgrd, lon, lat):
    longrd_flat, latgrd_flat = longrd.flatten(), latgrd.flatten()

    idx = np.argmin([haversine([ln, lon], [lt, lat])[0] for ln, lt in zip(longrd_flat, latgrd_flat)])

    return longrd_flat[idx], latgrd_flat[idx], idx
