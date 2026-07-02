import numpy as np
import xarray as xr

__all__ = ['Grid', 'derivative']

class Grid(object):
    def __init__(self, dim_mapping, proj='wgs84', boundary='extend'):
        if boundary == 'extend':
            boundary = {dim: ('extend', 'extend') for dim in dim_mapping}
        
        self.dmap   = dim_mapping
        self.bounds = boundary
        self.proj   = proj
        
        # Lazy cache for the per-grid metric scales. Keyed by a fingerprint
        # of the (lat, lon) coord arrays so reusing the same Grid on a
        # different DataArray with different coords rebuilds correctly.
        self._scale_cache = {}
    
    def __repr__(self):
        typ = ("Name,\tdimension,\tboundaries (l-r)\t=>\t"
               "'{:s}' coords\n".format(self.proj))
        out = ['{:>1s}\t{:>10s}\t{:>10s}\n'.format(str(dim), str(name), str(self.bounds[dim]))
               for dim, name in self.dmap.items()]
        return typ + ''.join(out)
    
    def gradient(self, var, dims=['X', 'Y'], boundary=None):
        """
        Spatial gradient components along each requested dimension.
        
        Metric scaling depends on self.proj:
          'wgs84' or 'geodetic' : WGS84 ellipsoidal distances via pyproj.Geod
          'spherical'           : spherical Earth (R cos phi for X, R for Y)
          'cartesian'           : coordinates already in meters
        
        Parameters
        ----------
        boundary : dict or None
            Optional per-call boundary override, e.g. {'X': ('periodic',)*2}.
        """
        scales = self._get_scales(var, dims)
        
        result = []
        for dim in dims:
            dim_name = self.dmap[dim]
            bc = boundary[dim] if boundary else self.bounds[dim]
            grd = derivative(var, dim_name, bc, scales[dim])
            result.append(grd)
        
        return result[0] if len(result) == 1 else result
    
    def divergence(self, u, v, boundary=None):
        """
        Horizontal divergence in flux form (retains the curvature term).
        """
        if self.proj == 'cartesian':
            dudx = self.gradient(u, dims=['X'], boundary=boundary)
            dvdy = self.gradient(v, dims=['Y'], boundary=boundary)
            return dudx + dvdy
        
        lat_name = self.dmap['Y']
        lon_name = self.dmap['X']
        deg2rad  = np.pi / 180
        cos_phi  = np.cos(u[lat_name] * deg2rad)
        
        scales = self._get_scales(u, ['X', 'Y'])
        
        bc_x = boundary['X'] if boundary else self.bounds['X']
        bc_y = boundary['Y'] if boundary else self.bounds['Y']
        
        du_dx    = derivative(u,           lon_name, bc_x, scales['X'])
        dvcos_dy = derivative(v * cos_phi, lat_name, bc_y, scales['Y'])
        
        return du_dx + dvcos_dy / cos_phi
    
    def curl(self, u, v, boundary=None):
        """
        Vertical component of horizontal curl in flux form.
        """
        if self.proj == 'cartesian':
            dvdx = self.gradient(v, dims=['X'], boundary=boundary)
            dudy = self.gradient(u, dims=['Y'], boundary=boundary)
            return dvdx - dudy
        
        lat_name = self.dmap['Y']
        lon_name = self.dmap['X']
        deg2rad  = np.pi / 180
        cos_phi  = np.cos(u[lat_name] * deg2rad)
        
        scales = self._get_scales(u, ['X', 'Y'])
        
        bc_x = boundary['X'] if boundary else self.bounds['X']
        bc_y = boundary['Y'] if boundary else self.bounds['Y']
        
        dv_dx    = derivative(v,           lon_name, bc_x, scales['X'])
        ducos_dy = derivative(u * cos_phi, lat_name, bc_y, scales['Y'])
        
        return dv_dx - ducos_dy / cos_phi
    
    def laplacian(self, var, boundary=None):
        """
        Horizontal Laplacian, computed as div(grad(f)) so the curvature
        term is included automatically.
        
        TODO: Currently the boundary condition is applied at every
        derivative call, meaning it gets applied twice when div(grad(.))
        is composed. This is harmless for 'extend' and 'periodic' (wrapping
        twice == wrapping once), but for any reflective or zero-flux
        boundary it will give the wrong behaviour near the boundary --
        the BC should only be applied at the outermost call. Restructure
        when adding such BCs.
        """
        gx, gy = self.gradient(var, dims=['X', 'Y'], boundary=boundary)
        return self.divergence(gx, gy, boundary=boundary)
    
    # ------------------------------------------------------------------
    # Internal: metric-scale caching
    # ------------------------------------------------------------------
    
    def _coord_fingerprint(self, var):
        """
        Build a small, hashable signature of the lat/lon coord values on
        `var`. Two DataArrays with the same coord values get the same
        fingerprint, regardless of identity.
        """
        parts = []
        for dim in ('Y', 'X'):
            name = self.dmap.get(dim)
            if name is None or name not in var.coords:
                parts.append((dim, None))
                continue
            arr = np.asarray(var.coords[name].values)
            # length + first + last + a midpoint is enough to distinguish
            # the grids you'd realistically use in practice and is O(1).
            mid = arr[len(arr) // 2] if arr.size else None
            parts.append((dim, arr.size, float(arr[0]) if arr.size else None,
                          float(arr[-1]) if arr.size else None,
                          float(mid) if mid is not None else None))
        return tuple(parts)
    
    def _get_scales(self, var, dims):
        """Return cached scales for this grid, building them on first use."""
        key = (self._coord_fingerprint(var), tuple(sorted(dims)))
        cached = self._scale_cache.get(key)
        if cached is not None:
            return cached
        
        scales = self._build_scales(var, dims, R=6371.2e3, deg2rad=np.pi / 180)
        self._scale_cache[key] = scales
        return scales
    
    def _build_scales(self, var, dims, R, deg2rad):
        """Return {dim: scale} where scale converts coord-units to meters."""
        if self.proj == 'spherical':
            scales = {}
            for dim in dims:
                if dim == 'Y':
                    scales[dim] = R * deg2rad
                elif dim == 'X':
                    if 'Y' in self.dmap and self.dmap['Y'] in var.dims:
                        cos = np.cos(var[self.dmap['Y']] * deg2rad)
                    else:
                        cos = 1
                    scales[dim] = R * cos * deg2rad
                else:
                    scales[dim] = 1
            return scales
        
        if self.proj in ('geodetic', 'wgs84'):
            from pyproj import Geod
            g = Geod(ellps='WGS84')
            
            lat_name = self.dmap['Y']
            lon_name = self.dmap['X']
            for c in (lat_name, lon_name):
                if c not in var.coords:
                    raise ValueError(
                        f"proj='{self.proj}' requires '{c}' as a coordinate on var.")
            
            lat1d = var[lat_name].values
            lon1d = var[lon_name].values
            lon2d, lat2d = np.meshgrid(lon1d, lat1d)
            
            _, _, dx_m = g.inv(lon2d[:, :-1], lat2d[:, :-1],
                               lon2d[:, 1:],  lat2d[:, 1:])
            _, _, dy_m = g.inv(lon2d[:-1, :], lat2d[:-1, :],
                               lon2d[1:, :],  lat2d[1:, :])
            
            dlon_deg = np.abs(np.diff(lon1d))[None, :]
            dlat_deg = np.abs(np.diff(lat1d))[:, None]
            
            sx = dx_m / dlon_deg
            sy = dy_m / dlat_deg
            sx = np.concatenate([sx, sx[:, -1:]], axis=1)
            sy = np.concatenate([sy, sy[-1:, :]], axis=0)
            
            scales = {
                'X': xr.DataArray(sx, dims=(lat_name, lon_name),
                                  coords={lat_name: lat1d, lon_name: lon1d}),
                'Y': xr.DataArray(sy, dims=(lat_name, lon_name),
                                  coords={lat_name: lat1d, lon_name: lon1d}),
            }
            for dim in dims:
                scales.setdefault(dim, 1)
            return scales
        
        return {dim: 1 for dim in dims}
    
    def clear_cache(self):
        """Drop the cached metric scales. Call this if you mutate the grid."""
        self._scale_cache = {}

def pad_boundary(var, dim, boundary='extend'):
    """
    Pad a DataArray along `dim` with one ghost point on each side.
    
    Parameters
    ----------
    boundary : str or (str, str)
        One of 'extend', 'periodic', 'reflect'. A single string is applied
        to both sides; a tuple specifies (left, right).
    """
    if isinstance(boundary, str):
        boundary = (boundary, boundary)
    
    mode_map = {'extend': 'edge', 'periodic': 'wrap', 'reflect': 'reflect'}
    
    out = var
    for side, shp in zip(boundary, [(1, 0), (0, 1)]):
        out = out.pad({dim: shp}, mode=mode_map[side])
    
    # Fix coord values on the padded edges by linear extrapolation,
    # so the spacing is well-defined for differentiate(). The actual data
    # in the ghost cells was already set correctly by xarray's pad.
    coord = out[dim].values.astype(float).copy()
    coord[0]  = coord[1]  - (coord[2]  - coord[1])
    coord[-1] = coord[-2] + (coord[-2] - coord[-3])
    out = out.assign_coords({dim: coord})
    return out


def derivative(var, dim, boundary='extend', scale=1):
    """First-order central derivative along `dim`, scaled to physical units."""
    pad_var = pad_boundary(var, dim, boundary)
    if pad_var.chunks is not None:
        pad_var = pad_var.chunk({dim: -1})
    deriv = pad_var.differentiate(dim).isel({dim: slice(1, -1)})
    return deriv / scale