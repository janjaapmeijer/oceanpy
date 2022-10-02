import xgcm

import numpy as np
from collections import defaultdict
from xgcm import Grid

__all__ = ['define_grid', 'horizontal_strain', 'horizontal_divergence',
           'relative_vorticity']

# def define_grid(ds, dims=('X', 'Y', 'Z'),
#                 distances=('dxt', 'dyt','dzt', 'dxu', 'dyu'),
#                 areas=('area_u', 'area_t')):
def define_grid(ds, dims, coords, distances, areas, periodic=False, boundary='extend'):

    '''
    Define staggered-grid in model

    Parameters
    ----------
    ds : Dataset


    Returns
    -------
    grid :

    Examples
    --------

    '''

    # TODO check if coordinates have attributes (hasattr) axis and c_grid_axis_shift
    # coords can be a dict or list or tuple
    # if type(coords) is dict:
    # try:
    #     shifts, axes = [], []
    #     for coord in coords:
    #         axes.append(ds[coord].attrs.axis)
    #         shifts.append(ds[coord].attrs.c_grid_axis_shift)
    # except:

    for name, shift in coords.items():
        for dim in dims:
            if dim.lower() in name.lower():
                ds.coords[name].attrs.update(axis=dim, c_grid_axis_shift=shift)

    metrics = defaultdict(list)
    for distance in distances:
        for dim in dims:
            if dim.lower() in distance.lower():
                metrics[(dim,)].append(distance)

    try:
        for area in sorted(areas):
            if len(dims) in (2, 3):
                name = dims if len(dims) == 2 else dims[:-1]
            else:
                raise NameError('dims should have length 2 or 3, but got %s' %len(dims))
            metrics[name].append(area)
    except NameError:
        pass

    return Grid(ds, metrics=metrics, periodic=periodic, boundary=boundary)

# def local_time_derivative():

def horizontal_strain(ds, grid):

    # normal strain
    dudx_t = grid.interp(grid.diff(ds.u, 'X', boundary='extend'), 'Y', boundary='extend') / ds.dxt
    dvdy_t = grid.interp(grid.diff(ds.v, 'Y', boundary='extend'), 'X', boundary='extend') / ds.dyt

    sn_t = dudx_t - dvdy_t
    sn_u = grid.interp(grid.interp(sn_t, 'X', boundary='extend'), 'Y', boundary='extend')

    # shear strain
    dvdx_t = grid.interp(grid.diff(ds.v, 'X', boundary='extend'), 'Y', boundary='extend') / ds.dxt
    dudy_t = grid.interp(grid.diff(ds.u, 'Y', boundary='extend'), 'X', boundary='extend') / ds.dyt

    ss_t = dvdx_t + dudy_t
    ss_u = grid.interp(grid.interp(ss_t, 'X', boundary='extend'), 'Y', boundary='extend')

    sigma = np.sqrt(sn_u**2 + ss_u**2)

    return sn_u, ss_u, sigma


def horizontal_divergence(ds, grid, vel_names=('u', 'v'), delta_names=('dx', 'dy')):

    # interpolated on t-cells
    dudx_t = grid.interp(grid.diff(ds[vel_names[0]], 'X', boundary='extend'), 'Y', boundary='extend') / ds[delta_names[0]]
    dvdy_t = grid.interp(grid.diff(ds[vel_names[1]], 'Y', boundary='extend'), 'X', boundary='extend') / ds[delta_names[1]]

    div_ht = dudx_t + dvdy_t

    # interpolated on u-cells
    div_hu = grid.interp(grid.interp(div_ht, 'X', boundary='extend'), 'Y', boundary='extend')

    div_hu.name = 'div_hu'
    div_hu.attrs['long_name'] = r'$\nabla_H \cdot {\bf u}$'

    return div_hu

def relative_vorticity(ds, grid, vel_names=('u', 'v'), delta_names=('dx', 'dy')):

    # interpolated on t-cells
    dvdx_t = grid.interp(grid.diff(ds[vel_names[1]], 'X', boundary='extend'), 'Y', boundary='extend') / ds[delta_names[0]]
    dudy_t = grid.interp(grid.diff(ds[vel_names[0]], 'Y', boundary='extend'), 'X', boundary='extend') / ds[delta_names[1]]

    zeta_t = dvdx_t - dudy_t

    # interpolated on u-cells
    zeta_u = grid.interp(grid.interp(zeta_t, 'X', boundary='extend'), 'Y', boundary='extend')

    zeta_u.name = 'zeta'

    return zeta_u


def vortex_stretching(ds, grid, vel_names=('u', 'v'), delta_names=('dx', 'dy')):

    # interpolate velocity gradients on t-cells
    dudx_t = grid.interp(grid.diff(ds[vel_names[0]], 'X', boundary='extend'), 'Y', boundary='extend') / ds[delta_names[0]]
    dvdy_t = grid.interp(grid.diff(ds[vel_names[1]], 'Y', boundary='extend'), 'X', boundary='extend') / ds[delta_names[1]]

    # horizontal divergence
    div_ht = dudx_t + dvdy_t

    # Coriolis parameter
    fcor,_ = xr.broadcast(f(ds.v.yu_ocean), ds.v.xu_ocean)

    # interpolate horizontal divergence to u-cells
    div_hu = grid.interp(grid.interp(div_ht, 'X', boundary='extend'), 'Y', boundary='extend')

    # vortex stretching term
    fdwdz = - fcor * div_hu

    # add attributes
    fdwdz.name = 'fdwdz'
    fdwdz.attrs['long_name'] = r'Vortex stretching, $-f\frac{\partial w}{\partial z}$'

    return fdwdz

def adv_relative_vorticity(ds, grid):

    # interpolate relative vorticity gradients on t-cells
    dzetadx_t = grid.interp(grid.diff(ds.zeta, 'X', boundary='extend'), 'Y', boundary='extend') / ds.dxt
    dzetady_t = grid.interp(grid.diff(ds.zeta, 'Y', boundary='extend'), 'X', boundary='extend') / ds.dyt

    # interpolate velocities to t-cells
    u_t = grid.interp(grid.interp(ds.u, 'X', boundary='extend'), 'Y', boundary='extend')
    v_t = grid.interp(grid.interp(ds.v, 'X', boundary='extend'), 'Y', boundary='extend')

    # advection of relative vorticity
    ugradzeta_t = (u_t * dzetadx_t) + (v_t * dzetady_t)

    # interpolated to u-cells
    ugradzeta = grid.interp(grid.interp(ugradzeta_t, 'X', boundary='extend'), 'Y', boundary='extend')

    # add attributes
    ugradzeta.name = 'ugradzeta'
    ugradzeta.attrs['long_name'] = r'Advection of relative vorticity, ${\bf u} \cdot \nabla\zeta$'

    return ugradzeta

def adv_planetary_vorticity(ds, grid):

    # Coriolis parameter
    fcor,_ = xr.broadcast(f(ds.yu_ocean), ds.xu_ocean)

    # meridional gradient interpolated on t-cells
    beta = grid.interp(grid.diff(fcor, 'Y', boundary='extend'), 'X', boundary='extend') / ds.dyt

    # interpolated on u-cells
    beta = grid.interp(grid.interp(beta, 'X', boundary='extend'), 'Y', boundary='extend')

    # advection of planetary vorticity
    betav = beta * ds.v

    # add attributes
    betav.name = 'betav'
    betav.attrs['long_name'] = 'Advection of planetary vorticity, $\beta v$'
    # beta.name = 'beta'
    # beta.attrs['long_name'] = 'Meridional gradient of the Coriolis parameter'

    return betav#, beta

def vorticity_tendency(ds):

    # local time derivative
    dzetadt = ds.zeta.chunk({'time': -1}).differentiate('time', datetime_unit='s')

    # add attributes
    dzetadt.name = 'dzetadt'
    dzetadt.attrs['long_name'] = 'Vorticity tendency, $\frac{\partial \zeta}{\partial t}$'

    return dzetadt

def bottom_pressure_torque(ds, grid):

    # Coriolis parameter
    fcor,_ = xr.broadcast(f(ds.yu_ocean), ds.xu_ocean)

    # topography gradient interpolated on u-cells
    dzbdx_u = grid.interp(grid.diff(ds.ht, 'X', boundary='extend'), 'Y', boundary='extend') / ds.dxu
    dzbdy_u = grid.interp(grid.diff(ds.ht, 'Y', boundary='extend'), 'X', boundary='extend') / ds.dyu

    # advection of relative vorticity
    last_cell = kmu_lim.copy()
    last_cell.values = last_cell.values.astype(int) - 3
    bpt = - fcor * ((ds.u.isel(st_ocean=last_cell) * dzbdx_u) + (ds.v.isel(st_ocean=kmu_lim) * dzbdy_u))

    return bpt

def stress_curl(ds, grid, stress_names=('tau_x', 'tau_y')):

    # interpolated on t-cells
    dtauydx_t = grid.interp(grid.diff(ds[stress_names[1]], 'X', boundary='extend'), 'Y', boundary='extend') / ds.dxt
    dtauxdy_t = grid.interp(grid.diff(ds[stress_names[0]], 'Y', boundary='extend'), 'X', boundary='extend') / ds.dyt

    curl_tau_t = dtauydx_t - dtauxdy_t

    # interpolated on u-cells
    curl_tau = grid.interp(grid.interp(curl_tau_t, 'X', boundary='extend'), 'Y', boundary='extend')

    curl_tau.name = 'curl_tau'

    return curl_tau
