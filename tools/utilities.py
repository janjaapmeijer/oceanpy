# from skimage import measure
import numpy as np
import pyproj

def contour_length(xr_ds, contour, var=None, timemean=True, time_sel=slice(None, None), lon_sel=slice(None, None), lat_sel=slice(None, None), transform=None):

    field = xr_ds[var] if var is not None else xr_ds.copy()

    # take the time mean field
    if timemean and time_sel != slice(None, None):
        field = field.sel(time=time_sel).mean(axis=0)
    elif timemean:
        field = field.mean(axis=0)
    elif not timemean:
        pass
    elif time_sel != slice(None, None):
        field = field.sel(time=time_sel)
    else:
        raise ValueError('Field can have a maximum number of 2 dimension but got %s', field.ndim)

    # select spatial area
    ctfield = field.copy()
    if lon_sel != slice(None, None):
        ctfield = ctfield.sel(longitude=lon_sel)
    if lat_sel != slice(None, None):
        ctfield = ctfield.sel(latitude=lat_sel)

    # find longest contour
    contours = measure.find_contours(ctfield, contour)
    contour = max(contours, key=lambda x: len(x))

    lon_ct, lat_ct = contour[:, 1], contour[:, 0]
    lon_ct = np.nanmin(ctfield.longitude) + np.unique(np.diff(ctfield.longitude)) * lon_ct
    lat_ct = np.nanmin(ctfield.latitude) + np.unique(np.diff(ctfield.latitude)) * lat_ct

    # transform polar coordinates to cartesian coordinates
    if transform is not None:
        WGS84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
        x_ct, y_ct = pyproj.transform(WGS84, transform, lon_ct, lat_ct)
    else:
        x_ct, y_ct = lon_ct.copy(), lat_ct.copy()

    #  calculate contour length in meters
    dx, dy = np.diff(x_ct), np.diff(y_ct)
    cl = np.sum(np.sqrt(dx**2 + dy**2))

    return cl, x_ct, y_ct, field
