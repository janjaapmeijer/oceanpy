import xgcm

from collections import defaultdict
from xgcm import Grid


# def define_grid(ds, dims=('X', 'Y', 'Z'),
#                 distances=('dxt', 'dyt','dzt', 'dxu', 'dyu'),
#                 areas=('area_u', 'area_t')):
def define_grid(ds, dims, coords, distances, areas, periodic=False):

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

    return Grid(ds, metrics=metrics, periodic=periodic)

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


def horizontal_divergence(ds, grid):

    # interpolated on t-cells
    dudx_t = grid.interp(grid.diff(ds.u, 'X', boundary='extend'), 'Y', boundary='extend') / ds.dxt
    dvdy_t = grid.interp(grid.diff(ds.v, 'Y', boundary='extend'), 'X', boundary='extend') / ds.dyt

    div_ht = dudx_t + dvdy_t

    # interpolated on u-cells
    div_hu = grid.interp(grid.interp(div_ht, 'X', boundary='extend'), 'Y', boundary='extend')

    return div_hu

# def relative_vorticity(ds, variables=('u', 'v'), coords=('xt', 'yt', 'xu', 'yu'),
#                        derivatives=('dxt', 'dyt','dzt', 'dxu', 'dyu')):
def relative_vorticity(ds, grid):

    # interpolated on t-cells
    dvdx_t = grid.interp(grid.diff(ds.v, 'X', boundary='extend'), 'Y', boundary='extend') / ds.dxt
    dudy_t = grid.interp(grid.diff(ds.u, 'Y', boundary='extend'), 'X', boundary='extend') / ds.dyt

    zeta_t = dvdx_t - dudy_t

    # interpolated on u-cells
    zeta_u = grid.interp(grid.interp(zeta_t, 'X', boundary='extend'), 'Y', boundary='extend')

    return zeta_u
