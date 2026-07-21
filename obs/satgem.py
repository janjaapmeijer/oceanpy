import gsw
import numpy as np
import xarray as xr
from scipy.interpolate import UnivariateSpline

from oceanpy.tools.netcdf import retrieve_attrs

__all__ = ['teos_from_insitu', 'connect_ssh_to_dynh', 'construct_gem', 'construct_satgem']

# ---------------------------------------------------------------------------
# Equation of state  →  SA, CT, rho
# ---------------------------------------------------------------------------

def teos_from_insitu(ds,
    variables=('t', 'SP', 'p'),
    coordinates=('longitude', 'latitude', 'pressure')): # dimensions=('profile', 'level'),

    """
    Compute Absolute Salinity and Conservative Temperature from T, S, p
    using TEOS-10 (gsw) on the original (station x pressure) grid.

    Parameters
    ----------
    ds : xr.Dataset
        Raw transect dataset with dims (station, pressure).
    variables : tuple of str
        Variable names for (temperature, salinity, pressure).
    dimensions : tuple of str
        Dimension names for (profile, level) axes.
    coordinates : tuple of str
        Coordinate names for (longitude, latitude, pressure).

    Returns
    -------
    xr.Dataset
        Dataset containing SA and CT on the original (station x pressure) grid.
    """
    
    t = ds[variables[0]]
    SP = ds[variables[1]]

    p_name = variables[2] if variables[2] in ds else coordinates[2]
    if p_name not in ds:
        raise KeyError(
            f"Pressure not found as variable '{variables[2]}' "
            f"or coordinate '{coordinates[2]}'.")
    p = ds[p_name]

    # Broadcast p to 2D if it is 1D along pressure
    if p.ndim == 1:
        p = p.broadcast_like(SP)

    lon = ds[coordinates[0]]
    lat = ds[coordinates[1]]
    # if lon.ndim == 1:
    #     lon = lon[:, np.newaxis]
    # if lat.ndim == 1:
    #     lat = lat[:, np.newaxis]

    SA = gsw.SA_from_SP(SP, p, lon, lat)
    SA.name = 'SA'
    CT = gsw.CT_from_t(SA, t, p)
    CT.name = 'CT'

    hydro = xr.merge([SA, CT])
    hydro.attrs.clear()
    hydro = hydro.assign_coords(ds.coords).assign_attrs(ds.attrs)
    variables = ['SA', 'CT']
    attrs = retrieve_attrs(variables)
    for var in variables:
        hydro[var] = hydro[var].assign_attrs(attrs[var])    
    
    return hydro

    # if density == 'rho':
    #     rho = gsw.rho(SA, CT, p)
    #     rho.name = 'rho'
    #     return xr.merge([SA, CT, rho])
    # elif density == 'sigma':
    #     sigma = gsw.sigma0(SA, CT)
    #     rho.name = 'rho'

# ---------------------------------------------------------------------------
# Connect SSH to GEM  →  SSH, TEOS10, Stream function (Dynamic Height)
# ---------------------------------------------------------------------------
def connect_ssh_to_dynh(
    ssh, streamfunc, 
    variables=('adt', 'deltaD', 'g'),
    coordinates=('time', 'longitude', 'latitude', 'pressure'),
    dimensions=('profile', 'pressure')):
    
    try:
        ssh = ssh[variables[0]]
    except AttributeError:
        pass

    # get SSH and DH at CTD locations
    ssh_ctd = ssh.sel(
        time=streamfunc[coordinates[0]], longitude=streamfunc[coordinates[1]], latitude=streamfunc[coordinates[2]], method='nearest').values
    p_idx = np.isfinite(streamfunc[variables[1]]).all(dimensions[0]).argmax()
    print('p_0 closest to surface: %s' % streamfunc[coordinates[3]].isel({dimensions[1]: p_idx}).values)
    D = streamfunc[variables[1]].isel({dimensions[1]: p_idx}) / streamfunc[variables[2]].isel({dimensions[1]: p_idx})

    # relation between SSH and DH (y = ax + b)
    mask = np.isfinite(ssh_ctd) & np.isfinite(D)
    coeffs = np.polyfit(ssh_ctd[mask], D[mask], deg=1)
    D_ssh = xr.apply_ufunc(np.polyval, coeffs, ssh, dask="parallelized", output_dtypes=[ssh.dtype])
    # valid = (D_ssh >= D.min()) & (D_ssh <= D.max())
    # D_ssh = D_ssh.where(valid)
        
    return coeffs, D

def construct_gem(
    hydro, D, window_frac=0.1, smooth_fac=2,
    variables=('CT', 'SA'),
    coordinates=('time', 'longitude', 'latitude', 'pressure'),
    dimensions=('profile', 'pressure')):

    # fit splines once over stations (D is static)
    # if minobs is None:
    npf = len(hydro[dimensions[0]])
    splines = {}
    for p in hydro[coordinates[3]].values:
        p = int(p)
        splines[p] = {}
        for var in variables:
            mask = np.isfinite(D.values) & np.isfinite(hydro[var].sel({coordinates[3]: p}).values)
            x_raw, y_raw = D.values[mask], hydro[var].sel({coordinates[3]: p}).values[mask]
            
            # sort by x for spline
            sort_idx = np.argsort(x_raw)
            x, y = x_raw[sort_idx], y_raw[sort_idx]
            n = len(x)

            if n >= 0.1 * npf:
                if var in ['CT', 'SA']:
                    splines[p][var] = UnivariateSpline(x, y)
                else:
                    window = max(5, int(round(n * window_frac)))
                    local_std = np.empty(n)
                    half = window // 2
                    for i in range(n):
                        lo, hi = max(0, i - half), min(n, i + half+1)
                        local_std[i] = np.std(y[lo:hi])
                    # guard against both zero and absurdly tiny local_std blowing up weights
                    floor = max(1e-6, 0.01 * np.nanstd(y))
                    local_std = np.clip(local_std, floor, None)  # avoid divide-by-zero
                    weight = 1.0 / local_std
                    
                    splines[p][var] = UnivariateSpline(x, y, w=weight, s=smooth_fac*len(weight))
            else:
                splines[p][var] = None

    return splines

def construct_satgem(
    ssh, splines, D_limits, coeffs, 
    variables=('CT', 'SA'), 
    coordinates=('time', 'longitude', 'latitude', 'pressure')):

    try:
        ssh = ssh['adt']
    except AttributeError:
        pass
    
    # find dynamic height values from relation (y=ax+b) with ADT
    D_ssh = xr.apply_ufunc(np.polyval, coeffs, ssh, dask="parallelized", output_dtypes=[ssh.dtype])
    valid = (D_ssh >= D_limits[0]) & (D_ssh <= D_limits[1])
    D_ssh = D_ssh.where(valid)

    # evaluate splines on full D_ssh grid (time, lat, lon) for each pressure level
    D_flat = D_ssh.values.flatten()  # (time * lat * lon,)
    pressure_coord = list(splines.keys())
    output_shape = D_ssh.shape + (len(pressure_coord),)  # (time, lat, lon, pressure)
    results = {var: np.full(output_shape, np.nan) for var in variables}
    for i, p in enumerate(pressure_coord):
        p = int(p)
        for var in variables:
            if splines[p][var] is not None:
                results[var][:,:,:,i] = splines[p][var](D_flat).reshape(D_ssh.shape)

    # assemble into Dataset
    dims = list(ssh.dims)
    coords = {dim: ssh[dim] for dim in dims}
    coords[coordinates[3]] = pressure_coord
    data_dict = {var: xr.DataArray(results[var], dims=dims+[coordinates[3]], coords=coords) for var in variables}

    # add ssh and dynamic height fields
    data_dict['D'] = xr.DataArray(D_ssh, dims=dims, coords={dim: ssh[dim] for dim in dims})
    satgem = xr.merge([xr.Dataset(data_dict), ssh])

    return satgem