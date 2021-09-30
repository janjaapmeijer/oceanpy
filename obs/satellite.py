import os

# import pyproj
import numpy as np

from netCDF4 import Dataset, num2date
from shutil import copyfile

from scipy.interpolate import griddata, UnivariateSpline
from stsci.convolve import boxcar
from gsw import f, grav

from ..tools import createNetCDF
from ..tools import haversine, rotatexy

__all__ = ['gradient_wind_from_ssh',
           'qg_from_ssh',
           ]

# from OceanPy.utilities import contour_length

# TODO: mask if any of the variables is nan
# import xarray as xr

# def gradient_balance_from_ssh(xr_ds, coord, variables=('adt', 'ugos', 'vgos'),
# dimensions=('longitude', 'latitude'), fcor=1e-4, gravity=9.81, transform=None, time=None):
#
#     # select which timestep
#     if time is not None:
#         xr_ds = xr_ds.sel(time=time)
#
#     # take Absolute Dynamic Topography and geostrophic velocities from SSH xarray
#     adt = xr_ds[variables[0]] if hasattr(xr_ds, variables[0]) else xr_ds.copy()
#     ugos = xr_ds[variables[1]] if hasattr(xr_ds, variables[1]) else None
#     vgos = xr_ds[variables[2]] if hasattr(xr_ds, variables[2]) else None
#
#     # check if field dimensions are 2-D
#     if adt.ndim != 2:
#         raise ValueError('Field can have a maximum number of 2 dimension but got %s', adt.ndim)
#
#     # transform polar in cartesian coordinate system
#     if transform is not None:
#         WGS84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
#         lnln, ltlt = np.meshgrid(xr_ds[dimensions[0]].data, xr_ds[dimensions[1]].data)
#         xx, yy = pyproj.transform(WGS84, transform, lnln, ltlt)
#         x, y = pyproj.transform(WGS84, transform, *coord)
#     else:
#         xx, yy = np.meshgrid(xr_ds[dimensions[0]].data, xr_ds[dimensions[1]].data)
#         x, y = coord
#     dx, dy = np.unique(np.diff(xr_ds[dimensions[0]]))[0], np.unique(np.diff(xr_ds[dimensions[1]]))[0]
#
#     # calculate geostrophy parameters
#     if transform is not None:
#         fcor = f(coord[1])
#         gravity = grav(coord[1], p=0)
#
#     # interpolate adt to coordinate location
#     points = np.array((xx.flatten(), yy.flatten())).T
#     adt_flat = adt.data.flatten()
#     adt_coord = griddata(points, adt_flat, (x, y))
#
#     # calculate geostrophic velocities at coordinate location
#     if ugos is None or vgos is None:
#         adtx = griddata(points, adt_flat, ([x - (dx / 2), x + (dx / 2)], [y, y]))
#         adty = griddata(points, adt_flat, ([x, x], [y - (dy / 2), y + (dy / 2)]))
#
#         dzetadx = np.diff(adtx) / dx
#         dzetady = np.diff(adty) / dy
#         ug = -(gravity / fcor) * dzetady
#         vg = (gravity / fcor) * dzetadx
#     else:
#         ugos_flat = ugos.data.flatten()
#         vgos_flat = vgos.data.flatten()
#         ug = griddata(points, ugos_flat, (x, y))
#         vg = griddata(points, vgos_flat, (x, y))
#     # if ug is positive, t is positive in x-direction and n in positive y direction
#     xpos = True if ug > 0 else False
#     ypos = True if vg > 0 else False
#
#     # find contour points close to interested data point
#     def strictly_increasing(L):
#         return all(i0 < i1 for i0, i1 in zip(L, L[1:]))
#     def strictly_decreasing(L):
#         return all(i0 > i1 for i0, i1 in zip(L, L[1:]))
#
#     coords_ct = contour_length(xr_ds=adt, contour=adt_coord, time_sel=time, timemean=False,
#                                lon_sel=slice(coord[0] - 2*dx, coord[0] + 2*dx),
#                                lat_sel=slice(coord[1] - 2*dy, coord[1] + 2*dy))[1:3]
#     if not (strictly_increasing(coords_ct[0]) | strictly_decreasing(coords_ct[0])):
#         # TODO: if transform: distance
#         idx = np.argsort(np.sqrt((coords_ct[0] - coord[0])**2 + (coords_ct[1] - coord[1])**2))[0:2]
#         coords_ct = (np.append(coords_ct[0][idx], coord[0]), np.append(coords_ct[1][idx], coord[1]))
#
#     # determine normal/ tangential resolution
#     x0 = (xr_ds[dimensions[0]].min() + xr_ds[dimensions[0]].max()) / 2
#     y0 = (xr_ds[dimensions[1]].min() + xr_ds[dimensions[1]].max()) / 2
#     if transform is not None:
#         x_ct, y_ct = pyproj.transform(WGS84, transform, *coords_ct)
#         dn = haversine([x0 - dx / 2, x0 + dx / 2], [y0 - dy / 2, y0 + dy / 2])[0][0]
#     else:
#         x_ct, y_ct = coords_ct
#         dn = np.sqrt(dx**2 + dy**2)
#
#     # calculate radius of curvature and orientation angle velocity vector
#     try:
#         fx = UnivariateSpline(x_ct, y_ct)
#     except ValueError:
#         # print('x is not increasing')
#         x_ct, y_ct = [lst for lst in zip(*sorted(zip(x_ct, y_ct), key=lambda pair: pair[0]))]
#         try:
#             fx = UnivariateSpline(x_ct, y_ct)
#             dydx = fx.derivative(1)(x)
#             d2ydx2 = fx.derivative(2)(x)
#         except:
#             dydx = np.gradient(y_ct)[1] / np.gradient(x_ct)[1]
#             d2ydx2 = dydx / np.gradient(x_ct)[1]
#     else:
#         dydx = fx.derivative(1)(x)
#         d2ydx2 = fx.derivative(2)(x)
#
#     Rcurv = (1 + dydx**2)**(3 / 2) / d2ydx2
#     orientation = np.arctan(dydx) if xpos else np.arctan(dydx) + np.pi
#
#     # determine locations of points normal to interested data point
#     xi = np.array([x - dn / 2, x + dn / 2])
#     yi = y * np.ones(len(xi))
#     ti, ni = zip(*rotatexy(x, y, xi, yi, orientation + (np.pi / 2)))
#
#     # interpolate ssh.adt to normal/ tangential points
#     adti = griddata(points, adt_flat, (ti, ni))
#
#     # geostrophic speed
#     dDdn = np.diff(adti) / dn
#     Vg = -(gravity / fcor) * dDdn
#
#     # gradient speed from Holten, 2004
#     hemisphere = 'SH' if fcor < 0 else 'NH'
#     root = np.sqrt(((fcor**2 * Rcurv**2) / 4) + (fcor * Rcurv * Vg))
# #     print(hemisphere, 'Vg', Vg, 'first term', -(fcor * Rcurv / 2), 'root', root)
# #     print('plus root', -fcor * Rcurv / 2 + root, 'min root', -fcor * Rcurv / 2 - root)
#     if hemisphere == 'NH':
#         if Rcurv > 0 and Vg > 0:
#             # regular low
#             V = -(fcor * Rcurv / 2) + root
#         elif Rcurv < 0 and Vg > 0:
#             # regular high
#             V = -(fcor * Rcurv / 2) - root
#         else:
#             V = np.nan
#
#     elif hemisphere == 'SH':
#         if Rcurv < 0 and Vg > 0:
#             # regular low
#             V = -(fcor * Rcurv / 2) + root
#         elif Rcurv > 0 and Vg > 0:
#             # regular high
#             V = -(fcor * Rcurv / 2) - root
#         # elif Rcurv < 0 and Vg < 0:
#         #     # TODO: not sure, anomalous high
#         #     V = -(fcor * Rcurv / 2) + root
#         # elif np.isnan(root) and np.isfinite(Vg):
#         #     V = Vg.copy()
#         #     print('geostrophic')
#         else:
#             V = np.nan
#
#     return V, Vg, orientation, ug, vg

varis = {
    'ugeos': ('surface_geostrophic_eastward_sea_water_velocity', 'f8'),
    'vgeos': ('surface_geostrophic_northward_sea_water_velocity', 'f8'),
    'ugrad': ('surface_eastward_sea_water_velocity', 'f8'),
    'vgrad': ('surface_northward_sea_water_velocity', 'f8'),
    'Vgeos': ('surface_geostrophic_sea_water_speed', 'f8'),
    'Vgrad': ('surface_gradient-wind_sea_water_speed', 'f8'),
    'ori': ('sea_water_velocity_to_direction', 'f8'),
    'zeta': ('ocean_relative_vorticity', 'f8'),
    'dzetadt': ('unsteady_relative_vorticity', 'f8'),
    'fdwdz': ('vortex_stretching', 'f8'),
    'betav': ('planetary_vorticity_advection', 'f8'),
    'ugradzeta': ('relative_vorticity_advection', 'f8'),
    'divag': ('divergence_of_velocity', 'f8'),
    'ow': ('okubo_weiss_parameter', 'f8'),
    'ow_norm': ('normalised_okubo_weiss_parameter', 'f8')
}

def interp(var, xx, yy):
    finite = np.isfinite(var).flatten()
    if not all(finite):
        points = np.array((xx.flatten(), yy.flatten())).T
        values = var.flatten()
        var = griddata(points[finite], values[finite], points).reshape(xx.shape)
    return var

def gradient_wind_from_ssh(input_file, output_file=None, variables=('adt', 'ugos', 'vgos'), group='gradient-wind',
                           dimensions=('time', 'latitude', 'longitude'), smooth=False, transform=None):

    """
    Gradient wind velocities as a function of sea surface height.

    Parameters
    ----------
    input_file : netcdf file

    Returns
    -------
    gradient_wind_velocities : dict or netcdf file

    """

    # load file and variables
    dsin = Dataset(input_file, 'r+')
    if output_file is not None and os.path.isfile(output_file):
        print('Output file %s already exists.' %os.path.basename(output_file))
        dsout = createNetCDF(output_file)
    elif output_file is not None and not os.path.isfile(output_file):
        copyfile(input_file, output_file)
        print('Output file %s, copied from input file %s.'
              %(os.path.basename(output_file), os.path.basename(input_file)))
        dsout = createNetCDF(output_file)

    # take Absolute Dynamic Topography from SSH xarray
    # adt = xr_ds[variables[0]] if hasattr(xr_ds, variables[0]) else xr_ds.copy()
    # ugeos = xr_ds[variables[1]].values if hasattr(xr_ds, variables[1]) else None
    # vgeos = xr_ds[variables[2]].values if hasattr(xr_ds, variables[2]) else None
    adt = dsin[variables[0]][:] if variables[0] in dsin.variables else dsin.copy()
    ugeos = dsin[variables[1]][:] if variables[1] in dsin.variables else None
    vgeos = dsin[variables[2]][:] if variables[2] in dsin.variables else None

    # load dimensions
    lat = dsin[dimensions[1]][:] if dimensions[1] in dsin.dimensions else None
    lon = dsin[dimensions[2]][:] if dimensions[2] in dsin.dimensions else None

    # transform polar to cartesian coordinate system
    if transform is not None:
        WGS84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
        lnln, ltlt = np.meshgrid(lon.data, lat.data)
        xx, yy = pyproj.transform(WGS84, transform, lnln, ltlt)
    else:
        #TODO: what if ltlt is not defined, it will get stuck at calculating coriolis and gravity
        xx, yy = np.meshgrid(lon.data, lat.data)

    # calculate coriolis force and gravity on grid
    shp = adt.shape
    gravity = grav(ltlt, p=0)
    fcor = f(ltlt)

    # detadx = np.ma.masked_all(shp)
    # detady = detadx.copy()
    # d2etadx2, d2etady2, d2etadxdy = detadx.copy(), detadx.copy(), detadx.copy()
    grid_point = (3, 3)
    kappa = np.ma.masked_all(shp)
    geostrophy = (ugeos is None) | (vgeos is None)
    if geostrophy:
        ugeos, vgeos = np.ma.masked_all(shp), np.ma.masked_all(shp)
    for it in range(dsin[dimensions[0]].size):

        detadx = np.gradient(adt[it,])[1] / np.gradient(xx)[1]
        detady = np.gradient(adt[it,])[0] / np.gradient(yy)[0]
        detadx = boxcar(interp(detadx, xx, yy), grid_point) if smooth else detadx
        detady = boxcar(interp(detady, xx, yy), grid_point) if smooth else detady

        d2etadxdy = np.gradient(detadx)[0] / np.gradient(yy)[0]
        d2etadxdy = boxcar(interp(d2etadxdy, xx, yy), grid_point) if smooth else d2etadxdy

        d2etadx2 = np.gradient(detadx)[1] / np.gradient(xx)[1]
        d2etady2 = np.gradient(detady)[0] / np.gradient(yy)[0]
        d2etadx2 = boxcar(interp(d2etadx2, xx, yy), grid_point) if smooth else d2etadx2
        d2etady2 = boxcar(interp(d2etady2, xx, yy), grid_point) if smooth else d2etady2

        kappa[it,] = (-(d2etadx2*detady**2) -(d2etady2*detadx**2) + (2*d2etadxdy*detadx*detady)) / (detadx**2 + detady**2)**(3/2)

        if geostrophy:
            ugeos[it,] = -(gravity / fcor) * detady
            vgeos[it,] = (gravity / fcor) * detadx

        del detadx, detady, d2etadxdy, d2etadx2, d2etady2

    xpos = ugeos < 0
    ypos = vgeos < 0
    orientation = np.arctan(vgeos / ugeos)
    orientation[xpos] = np.arctan(vgeos[xpos] / ugeos[xpos]) + np.pi
    orientation[xpos & ypos] = np.arctan(vgeos[xpos & ypos] / ugeos[xpos & ypos]) - np.pi
    Vgeos = np.sqrt(ugeos**2 + vgeos**2)

    # kappa = (-(d2etadx2*detady**2) -(d2etady2*detadx**2) + (2*d2etadxdy*detadx*detady)) / (detadx**2 + detady**2)**(3/2)
    if adt.ndim != 2:
        # gravity = np.broadcast_to(gravity, shp)
        fcor = np.broadcast_to(fcor, shp)
    Rcurv = 1 / kappa
    root = np.sqrt(((fcor**2 * Rcurv**2) / 4) + (fcor * Rcurv * Vgeos))

    Vgrad = np.ma.masked_all(shp).flatten()
    fcor, Vgeos, Rcurv, root = fcor.flatten(), Vgeos.flatten(), Rcurv.flatten(), root.flatten()
    for i in range(len(Vgrad)):
        # Northern Hemisphere
        if fcor[i] >= 0:
            if (Rcurv[i] < 0) & (Vgeos[i] > 0):
                Vgrad[i] = -(fcor[i] * Rcurv[i] / 2) - root[i]
            elif (Rcurv[i] > 0) & (Vgeos[i] > 0):
                Vgrad[i] = -(fcor[i] * Rcurv[i] / 2) + root[i]
            else:
                Vgrad[i] = np.nan
        # Southern Hemisphere
        elif fcor[i] < 0:
            if (Rcurv[i] < 0) & (Vgeos[i] > 0):
                Vgrad[i] = -(fcor[i] * Rcurv[i] / 2) + root[i]
            elif (Rcurv[i] > 0) & (Vgeos[i] > 0):
                Vgrad[i] = -(fcor[i] * Rcurv[i] / 2) - root[i]
            else:
                Vgrad[i] = np.nan

    data = {}
    data['ugeos'], data['vgeos'], data['ori'] = ugeos, vgeos, orientation
    data['Vgrad'], data['Vgeos'] = Vgrad.reshape(shp), Vgeos.reshape(shp)
    data['ugrad'], data['vgrad'] = data['Vgrad'] * np.cos(orientation), data['Vgrad'] * np.sin(orientation)
    # new_variable = {
    # 'ugeos': ugeos, 'vgeos': vgeos, 'Vgeos': Vgeos,
    # 'ugrad': ugrad, 'vgrad': vgrad, 'Vgrad': Vgrad
    # }

    new_variables = {}
    for var in data.keys():
        new_variables['/%s/%s' %(group, var)] = varis[var] + (dimensions, ) + (data[var],)

    # save data in netcdf file using OceanPy's createNetCDF class
    if output_file is not None:

        # create group
        gw = dsout.dataset.createGroup(group)

        # create dimensions and Coordinates
        for name, dimension in dsin.dimensions.items():
            gw.createDimension(name, (dimension.size if not dimension.isunlimited() else None))
            if name in dimensions:
                if name == 'time':
                    values = num2date(dsin[name][:], units=dsin[name].units, calendar=dsin[name].calendar)
                else:
                    values = dsin[name][:]
                new_variables['/%s/%s' %(group, name)] = (name, 'f8') + (name, values)

        # create variables
        dsout.create_vars(new_variables)

    return new_variables if output_file is None else print('New variables %s, stored in group %s, of the output file.'
          % (', '.join([var for var in data.keys() if var in dsout.dataset[group].variables.keys()]), group))

    # variables = {'Vgrad': Vgrad}
    # for key, values in variables.items():
    #     xr_ds.assign(key = (dimensions, values))


    # xr_ds_new = xr.Dataset(data_vars={'Vgrad': (dimensions, Vgrad),
    #                                   'Vgeos': (dimensions, Vgeos),
    #                                   'orientation': (dimensions, orientation),
    #                                   'ugrad': (dimensions, ugrad),
    #                                   'vgrad': (dimensions, vgrad),
    #                                   'ugeos': (dimensions, ugeos),
    #                                   'vgeos': (dimensions, vgeos)},
    #                        coords={dimensions[0]: xr_ds[dimensions[0]],
    #                                dimensions[1]: xr_ds[dimensions[1]],
    #                                dimensions[2]: xr_ds[dimensions[2]]})

    # return dsout if output_file is not None else new_variables# xr_ds_new

def qg_from_ssh(input_file, output_file=None, group='quasi-geostrophy',
                dimensions=('time', 'latitude', 'longitude'), smooth=False, transform=None):

    """
    Calculate the quasi-geostrophic vorticity terms from gradient wind velocities.

    Parameters
    ----------
    input_file : netcdf-file

    Returns
    -------

    """

    # load file and variables
    groups = ['gradient-wind', 'gw']
    groupin = [grp for grp in Dataset(input_file).groups.keys() if grp.lower() in groups]
    if groupin:
        dsin = Dataset(input_file)[groupin[0]]
    else:
        print('Gradient wind variables not found in input file.')

    # dsin = Dataset(input_file, 'r+')
    if output_file is not None and os.path.isfile(output_file):
        print('Output file %s already exists.' %os.path.basename(output_file))
        dsout = createNetCDF(output_file)
    elif output_file is not None and not os.path.isfile(output_file):
        copyfile(input_file, output_file)
        print('Output file %s, copied from input file %s.'
              %(os.path.basename(output_file), os.path.basename(input_file)))
        dsout = createNetCDF(output_file)

    # load dimensions
    lat = dsin[dimensions[1]][:] if dimensions[1] in dsin.dimensions else None
    lon = dsin[dimensions[2]][:] if dimensions[2] in dsin.dimensions else None

    # transform polar to cartesian coordinate system
    if transform is not None:
        WGS84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
        lnln, ltlt = np.meshgrid(lon.data, lat.data)
        xx, yy = pyproj.transform(WGS84, transform, lnln, ltlt)
    else:
        #TODO: what if ltlt is not defined, it will get stuck at calculating coriolis and gravity
        xx, yy = np.meshgrid(lon.data, lat.data)

    # calculate Coriolis parameters
    fcor = f(ltlt)
    beta = np.gradient(fcor)[0] / np.gradient(yy)[0]

    # calculate ageostrophic components from gradient wind velocities
    ugrad, vgrad = dsin['ugrad'][:], dsin['vgrad'][:]
    uag = ugrad - dsin['ugeos'][:]
    vag = vgrad - dsin['vgeos'][:]

    smooth = True
    grid_point = (3, 3)

    zeta = np.ma.masked_all(ugrad.shape)
    divgeos, divag = zeta.copy(), zeta.copy()
    fdwdz, betav, ugradzeta = zeta.copy(), zeta.copy(), zeta.copy()
    ow, ow_norm = zeta.copy(), zeta.copy()
    for t in range(len(dsin['time'])):

        # (gradients of) relative vorticity
        dvdx = np.gradient(vgrad[t,])[1] / np.gradient(xx)[1]
        dvdx = boxcar(interp(dvdx, lnln, ltlt), grid_point) if smooth else dvdx
        dudy = np.gradient(ugrad[t,])[0] / np.gradient(yy)[0]
        dudy = boxcar(interp(dudy, lnln, ltlt), grid_point) if smooth else dudy

        zeta[t,] = dvdx - dudy

        dzetadx = boxcar(np.gradient(zeta[t,])[1] / np.gradient(xx)[1], grid_point)
        dzetady = boxcar(np.gradient(zeta[t,])[0] / np.gradient(yy)[0], grid_point)

        # normal and shear components of strain and Okubo-Weiss parameter
        # dudx = np.gradient(ugrad[t,])[1] / np.gradient(xx)[1]
        # dudx = boxcar(interp(dudx, lnln, ltlt), grid_point) if smooth else dudx
        # dvdy = np.gradient(vgrad[t,])[0] / np.gradient(yy)[0]
        # dvdy = boxcar(interp(dvdy, lnln, ltlt), grid_point) if smooth else dvdy

        # Okubo-Weiss parameter
        # sn = dudx - dvdy
        # ss = dvdx + dudy
        #
        # ow[t,] = sn**2 + ss**2 - zeta[t,]**2
        # W0 = 0.2 * np.nanstd(ow[t,])
        # ow_norm[t,] = ow[t,] / W0
        # ow[t,] = ow[t,] / 4

        # divergence of velocity field
        duagdx = np.gradient(uag[t,])[1] / np.gradient(xx)[1]
        duagdx = boxcar(interp(duagdx, lnln, ltlt), grid_point) if smooth else duagdx
        dvagdy = np.gradient(vag[t,])[0] / np.gradient(yy)[0]
        dvagdy = boxcar(interp(dvagdy, lnln, ltlt), grid_point) if smooth else dvagdy

        divag[t,] = duagdx + dvagdy

        # calculate vortivity budget terms
        fdwdz[t,] = - fcor * (duagdx + dvagdy)# - (beta * gw.vgeos[t,])
        betav[t,] = beta * vgrad[t,]
        ugradzeta[t,] = (ugrad[t,] * dzetadx) + (vgrad[t,] * dzetady)

        del dvdx, dudy, duagdx, dvagdy, dzetadx, dzetady # , dudx, dvdy, sn, ss

    # local time derivative of relative vorticity
    dt = np.gradient(dsin['time'][:]).astype('timedelta64[s]')
    dzetadt = np.gradient(zeta)[0] / np.unique(dt).astype('float')

    # store data in dictionary
    data = {}
    data['zeta'], data['dzetadt'], data['ugradzeta'] = zeta, dzetadt, ugradzeta
    data['fdwdz'], data['betav'], data['divag'] = fdwdz, betav, divag

    new_variables = {}
    for var in data.keys():
        new_variables['/%s/%s' %(group, var)] = varis[var] + (dimensions, ) + (data[var],)

    # save data in netcdf file using OceanPy's createNetCDF class
    if output_file is not None:

        # create group
        qgvb = dsout.dataset.createGroup(group)

        # create dimensions and Coordinates
        for name, dimension in dsin.dimensions.items():
            qgvb.createDimension(name, (dimension.size if not dimension.isunlimited() else None))
            if name in dimensions:
                if name == 'time':
                    try:
                        values = num2date(dsin[name][:], units=dsin[name].units, calendar=dsin[name].calendar)
                    except AttributeError:
                        dsroot = Dataset(input_file)
                        values = num2date(dsroot[name][:], units=dsroot[name].units, calendar=dsroot[name].calendar)
                else:
                    values = dsin[name][:]
                new_variables['/%s/%s' %(group, name)] = (name, 'f8') + (name, values)

        # create variables
        dsout.create_vars(new_variables)

    return new_variables if output_file is None else print('New variables %s, stored in group %s, of the output file.'
          % (', '.join([var for var in data.keys() if var in dsout.dataset[group].variables.keys()]), group))
