import xarray as xr
import numpy as np
from gsw import Nsquared, z_from_p
from scipy.integrate import cumulative_trapezoid

__all__ = ['buoyancy_frequency', 'regrid_to_depth', 'ageostrophic_velocity_from_omega']

def buoyancy_frequency(satgem, lon_name='longitude', lat_name='latitude', z_name='pressure'):
    """
    N² = -g/rho0 * d(rho)/dz       buoyancy frequency squared (s^-2)

    Parameters
    ----------
    rho  : xr.DataArray (z, x)   in-situ density (kg/m^3)
    g    : float                 gravitational acceleration (m/s^2)
    rho0 : float                 reference density (kg/m^3)

    Returns
    -------
    xr.Dataset with N2
    """

    N2, p_mid = xr.apply_ufunc(
        Nsquared,
        satgem.SA, satgem.CT, satgem.pressure, satgem.latitude,
        input_core_dims=[[z_name], [z_name], [z_name], []],
        output_core_dims=[[z_name+'_mid'], [z_name+'_mid']],
        vectorize=True, dask='parallelized',
        output_dtypes=[satgem.SA.dtype, satgem.pressure.dtype],
        dask_gufunc_kwargs={'output_sizes': {z_name+'_mid': satgem.sizes[z_name] - 1}})
    
    pressure_mid = (satgem.pressure[1:].values + satgem.pressure[:-1].values)/2
    N2 = N2.assign_coords(pressure_mid=pressure_mid)
    N2 = N2.interp(pressure_mid=satgem.pressure.values).rename({z_name+'_mid': z_name})
    N2.attrs = {'long_name': 'Buoyancy frequency squared', 'units': 's-2'}

    return N2

def regrid_to_depth(ds, ssh, deltaD, gravity, depth_grd, variables=None):
    """
    Transform a dataset from pressure to depth coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain variables 'CT', 'SA', 'adt' and dims/coords
        'time', 'latitude', 'longitude', 'pressure'.
    gravity : float
        Gravitational acceleration (m/s^2) used to compute surface geopotential.
    depth_grd : array-like, optional
        Target depth levels (negative downward, in metres).
        Defaults to np.arange(-1500, 0, 10).

    Returns
    -------
    xr.Dataset
        Dataset with CT, SA (and pressure as a variable) on dims
        (time, latitude, longitude, depth).
    """
    depth_grd = np.asarray(depth_grd, dtype=float)
    if variables is None:
        variables = [var for var in list(ds.variables.keys()) if var not in list(ds.coords.keys())]
    
    # --- 1. Compute depth from pressure (your existing logic) ---
    # geopotential at p_0 surface
    # p_0 = rho_0 * g * eta
    # surface geopotential
    Phi = gravity * ssh.adt
    
    depth = xr.apply_ufunc(z_from_p, ds.pressure, ds.latitude, deltaD, Phi, dask="parallelized", output_dtypes=[float],)
    
    # depth dims are broadcast across (time, latitude, longitude, pressure)
    depth = depth.transpose("time", "latitude", "longitude", "pressure")

    # --- 2. 1D interpolation along pressure for each (t, lat, lon) column ---
    def _interp_column(var_col, depth_col, target):
        # np.interp requires strictly increasing x, so flip.
        order = np.argsort(depth_col)
        return np.interp(target, depth_col[order], var_col[order], left=np.nan, right=np.nan)

    def _to_depth(var):
        return xr.apply_ufunc(_interp_column, var, depth,
            input_core_dims=[["pressure"], ["pressure"]], output_core_dims=[["depth"]], exclude_dims={"pressure"},
            vectorize=True, kwargs={"target": depth_grd}, output_dtypes=[float], dask="parallelized", dask_gufunc_kwargs={"output_sizes": {"depth": len(depth_grd)}})

    data_vars = {}
    for var in variables:
        data_vars[var] = _to_depth(ds[var])

    # Pressure as a data variable on the new grid
    if 'pressure' in ds.coords:    
        pressure_b = ds.pressure.broadcast_like(depth)
        p_d = _to_depth(pressure_b)
        data_vars['p'] = p_d

    # --- 3. Return data in xarray.Dataset ---
    out = xr.Dataset( data_vars,
        coords={"time": ds.time, "latitude": ds.latitude, "longitude": ds.longitude, "depth": depth_grd},)
    out["depth"].attrs.update(units="m", positive="up", long_name="depth (negative downward)")
    if 'p' in data_vars.keys():
        out["p"].attrs.update(units="dbar", long_name="sea water pressure")    
    
    return out

def ageostrophic_velocity_from_omega(w, Qx, Qy, N2, f, finitediff=None, bc_level="bottom"):

    # ------------------------------------------------------------------
    # 0. Validate inputs
    # ------------------------------------------------------------------
    if finitediff is None:
        dim_mapping = {"Z": "depth", "Y": "latitude", "X": "longitude"}
        BCs = {'X': ('extend', 'extend'),
               'Y': ('extend', 'extend'),
               'Z': ('extend', 'extend')}
        coords='lat-lon'
        fd = FiniteDiff(dim_mapping, BCs=BCs, coords=coords, fill=0)
    else:
        fd = finitediff

        keys = fd.dmap.keys()
        for key in keys:
            if key not in keys:
                raise ValueError(f"dim_mapping must contain '{key}'. Got: {keys}")
     
    zdim = fd.dmap['Z']
    if zdim not in w.dims:
        raise ValueError(f"Z dimension '{zdim}' not found in w.dims = {w.dims}")
 
    if bc_level not in ("bottom", "top"):
        raise ValueError(f"bc_level must be 'bottom' or 'top', got '{bc_level}'")
    
    # ------------------------------------------------------------------
    # 2. Horizontal gradients of w: dw/dx and dw/dy
    #    FiniteDiff.grad() in 'cartesian' mode returns true physical
    #    gradients (dw/dx in s^-1 for w in m/s and x in m).
    #    In 'lat-lon' mode it scales by (R * cos(lat) * deg2rad)^-1.
    # ------------------------------------------------------------------
    dw_dx, dw_dy = fd.grad(w, dims=["X", "Y"])
 
    # ------------------------------------------------------------------
    # 3. Vertical gradients of u_ag and v_ag
    #    d(u_ag)/dz = (N^2 * dw/dx - 2*Q1) / f
    #    d(v_ag)/dz = (N^2 * dw/dy - 2*Q2) / f
    # ------------------------------------------------------------------
    duag_dz = ((N2 * dw_dx) - (2.0 * Qx)) / f**2
    dvag_dz = ((N2 * dw_dy) - (2.0 * Qy)) / f**2
 
    # ------------------------------------------------------------------
    # 4. Vertical integration with the trapezoidal rule
    #    scipy's cumulative_trapezoid(initial=0) sets the first level to
    #    zero.  For bc_level='bottom' we reverse the Z axis so that the
    #    deepest level is anchored to zero, then reverse back.
    # ------------------------------------------------------------------
    def _cumtrapz_xr(var, zdim, bc_level):
        """
        Integrate dv/dz along zdim using the cumulative trapezoidal rule.
 
        Parameters
        ----------
        dv_dz_da : xr.DataArray
            Integrand — vertical gradient of u_ag or v_ag.
        zdim : str
            Name of the vertical dimension to integrate along.
        bc_level : str
            'bottom' anchors the deepest index to zero;
            'top' anchors the shallowest index to zero.
        """
        # Move Z to leading axis for numpy operations
        da  = var.transpose(zdim, ...)
        arr = da.values        # shape: (nz, ...)
        z   = var[zdim].values  # 1-D coordinate array
        
        if bc_level == "bottom":
            # cumulative_trapezoid(initial=0) anchors index 0 to zero.
            # For ocean grids stored deepest-first (z[0] = most negative
            # depth), this directly gives u_ag = 0 at the bottom.
            integrated = cumulative_trapezoid(arr, x=z, axis=0, initial=0)
        else:  # 'top' — anchor the shallowest level (index -1) to zero
            # Reverse Z, integrate (anchoring reversed index 0 = original
            # index -1 = surface), then reverse back.
            integ_r = cumulative_trapezoid(arr[::-1], x=z[::-1], axis=0, initial=0)
            integrated = -integ_r[::-1]
 
        # Re-wrap preserving all coordinates and restore dimension order
        result = xr.DataArray(integrated, coords=da.coords, dims=da.dims)
        return result
 
    u_ag = _cumtrapz_xr(duag_dz, zdim, bc_level).transpose(*w.dims)
    v_ag = _cumtrapz_xr(dvag_dz, zdim, bc_level).transpose(*w.dims)

    ds_out = xr.Dataset({"u_ag": u_ag, "v_ag": v_ag, "w_QG": w})
 
    return ds_out