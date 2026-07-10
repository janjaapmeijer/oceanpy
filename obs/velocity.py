"""
Classes
-------
StreamFunction
    Wraps a scalar field (SSH or dynamic height anomaly) and a Grid.
    Computes and caches spatial derivatives, curvature, and flow
    orientation using the Grid's metric throughout.
 
GeostrophicVelocity
    Geostrophic velocity from SSH (adt), dynamic height anomaly (deltaD),
    or both combined.  The mode is determined by which inputs are provided:
 
      adt only   → barotropic surface geostrophic velocity
      deltaD only → baroclinic shear only (no surface reference)
      adt + deltaD → combined: barotropic + baroclinic shear
 
GradientWindVelocity
    Gradient wind correction applied on top of GeostrophicVelocity.
    Accepts the same adt / deltaD inputs and constructs the geostrophic
    velocity internally.  Solves the gradient wind equation for each
    hemisphere and curvature sign (Holton, 2004), falling back to
    geostrophic speed where the discriminant is negative or curvature
    is zero.
"""
 
from __future__ import annotations
 
import numpy as np
import xarray as xr
from distributed import get_client
from oceanpy.tools.netcdf import retrieve_attrs

from gsw import grav, geo_strf_dyn_height, f as coriolis
 
__all__ = [
    'StreamFunction',
    'GeostrophicVelocity',
    'GradientWindVelocity',
    'streamfunction',
]
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _broadcast_fcor(lat: xr.DataArray, like: xr.DataArray) -> xr.DataArray:
    """
    Coriolis parameter broadcast to the shape of `like`.
    """

    fcor = xr.DataArray(coriolis(lat.values), coords=lat.coords, dims=lat.dims)
    fcor, _ = xr.broadcast(fcor, like)
    return fcor
 
 
def _broadcast_gravity(lat: xr.DataArray, like: xr.DataArray) -> xr.DataArray:
    """Surface gravity broadcast to the shape of `like`."""
    g = xr.DataArray(grav(lat.values, p=0), coords=lat.coords, dims=lat.dims)
    g, _ = xr.broadcast(g, like)
    return g
 
 
# def _smooth(field: xr.DataArray, method: str | None, window: int) -> xr.DataArray:
#     """
#     Optional 2-D smoothing on a DataArray.
#     Works on the last two dimensions so it is safe inside apply_ufunc
#     and on full (time, lat, lon) arrays alike.
#     """
#     if method is None:
#         return field
#     from scipy.ndimage import uniform_filter, gaussian_filter
#     fn = uniform_filter if method == 'boxcar' else gaussian_filter
#     return xr.apply_ufunc(
#         fn, field, kwargs={'size': window},
#         dask='parallelized', output_dtypes=[field.dtype],
#     )

def _smooth(field: xr.DataArray, method: str | None, window: int) -> xr.DataArray:
    """
    NaN-aware 2-D smoothing on a DataArray.
 
    Both scipy filters propagate NaNs across the entire kernel window,
    which blanks out large regions near land/data boundaries.
    Fix: replace NaNs with zero, smooth field and validity mask
    separately with the same kernel, then divide.  Equivalent to a
    weighted average where NaN cells contribute zero weight.
    Result is NaN only where the *entire* kernel window was NaN.
 
    Parameters
    ----------
    method : 'boxcar' or 'gaussian'
        boxcar   -> scipy.ndimage.uniform_filter  (kernel size = window)
        gaussian -> scipy.ndimage.gaussian_filter (sigma = window)
    window : int
        For boxcar:   kernel width in grid cells.
        For gaussian: standard deviation in grid cells.
    """
    if method is None:
        return field
 
    from scipy.ndimage import uniform_filter, gaussian_filter
 
    is_gaussian = method == 'gaussian'
    fn  = gaussian_filter if is_gaussian else uniform_filter
    kw  = {'sigma': window} if is_gaussian else {'size': window}
 
    def _nan_aware(arr):
        mask        = np.isfinite(arr).astype(float)
        filled      = np.where(np.isfinite(arr), arr, 0.0)
        smooth_data = fn(filled, **kw)
        smooth_mask = fn(mask,   **kw)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(smooth_mask > 0, smooth_data / smooth_mask, np.nan)
 
    return xr.apply_ufunc(
        _nan_aware, field,
        dask='parallelized', output_dtypes=[field.dtype],
    )

 
 
# ---------------------------------------------------------------------------
# StreamFunction
# ---------------------------------------------------------------------------
 
class StreamFunction:
    """
    Scalar stream-function field with metric-aware spatial derivatives.
 
    Parameters
    ----------
    field : xr.DataArray
        The scalar field ψ.  Units: m (SSH/ADT) or m² s⁻² (ΔD).
    grid : Grid
        An initialised Grid instance from grid.py.
    smooth : None | str | dict
        Optional smoothing applied to the first derivatives before any
        further computation.  Accepts ``'boxcar'``, ``'gaussian'``,
        or ``{'boxcar': window}`` / ``{'gaussian': window}``.
    """
 
    def __init__(self, field: xr.DataArray, grid, smooth=None):
        self.field        = field
        self.grid         = grid
        self._smooth_spec = self._parse_smooth(smooth)
        self._dfdx        = None   # lazy
        self._dfdy        = None
 
    # ------------------------------------------------------------------
    # Public properties — all lazy, computed on first access
    # ------------------------------------------------------------------
 
    @property
    def dfdx(self) -> xr.DataArray:
        """∂ψ/∂x in physical units per metre (eastward positive)."""
        if self._dfdx is None:
            self._compute_gradients()
        return self._dfdx
 
    @property
    def dfdy(self) -> xr.DataArray:
        """∂ψ/∂y in physical units per metre (northward positive)."""
        if self._dfdy is None:
            self._compute_gradients()
        return self._dfdy
 
    @property
    def speed(self) -> xr.DataArray:
        """Magnitude of the horizontal gradient |∇ψ|."""
        return np.sqrt(self.dfdx**2 + self.dfdy**2)
 
    @property
    def curvature(self) -> xr.DataArray:
        """
        Signed curvature κ of the streamlines (m⁻¹).
 
            κ = [-(∂²ψ/∂x²)(∂ψ/∂y)² - (∂²ψ/∂y²)(∂ψ/∂x)²
                  + 2(∂²ψ/∂x∂y)(∂ψ/∂x)(∂ψ/∂y)] / |∇ψ|³
 
        Masked where |∇ψ| ≈ 0 (stagnation points).
        """
        d2fdx2  = self.grid.gradient(self.dfdx, dims=['X'])
        d2fdy2  = self.grid.gradient(self.dfdy, dims=['Y'])
        d2fdxdy = self.grid.gradient(self.dfdx, dims=['Y'])
 
        px, py = self.dfdx, self.dfdy
        denom  = (px**2 + py**2) ** (3/2)
 
        kappa = (- d2fdx2  * py**2
                 - d2fdy2  * px**2
                 + 2 * d2fdxdy * px * py) / denom
 
        return kappa.where(denom > 0)
 
    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
 
    def _compute_gradients(self):
        method, window = self._smooth_spec

        # Identify the valid CTD coverage mask before any filling
        valid = np.isfinite(self.field)

        # Fill NaNs at the boundary with zero — we will mask them back
        # afterward. Zero is safe here because we're not extrapolating
        # into the NaN region — we only use the result where valid is True,
        # expanded inward by one cell to exclude stencil-contaminated edges.
        field_filled = self.field.fillna(0.0)
        
        gx, gy = self.grid.gradient(field_filled, dims=['X', 'Y'])
    
        # Erode the valid mask by 1 cell in each direction to exclude
        # any derivative point whose stencil touched a NaN neighbour.
        # rolling(...).min() acts as a morphological erosion:
        #   if any cell in the 3-point window is 0 (was NaN), result is 0.
        lon_dim = self.grid.dmap['X']
        lat_dim = self.grid.dmap['Y']
        eroded  = (valid.astype(float)
                .rolling({lon_dim: 3}, center=True, min_periods=3).min()
                .rolling({lat_dim: 3}, center=True, min_periods=3).min()
                .fillna(0)
                .astype(bool))

        self._dfdx = _smooth(gx.where(eroded), method, window)
        self._dfdy = _smooth(gy.where(eroded), method, window)
 
    @staticmethod
    def _parse_smooth(smooth) -> tuple[str | None, int]:
        if smooth is None or smooth is False:
            return None, 3
        methods = ('boxcar', 'gaussian')
        if isinstance(smooth, dict):
            method, window = next(iter(smooth.items()))
            if method not in methods:
                raise ValueError(
                    f"smooth method must be one of {methods}, got '{method}'")
            if not isinstance(window, int):
                raise TypeError(
                    f"smooth window must be int, got {type(window)}")
            return method, window
        if isinstance(smooth, str):
            if smooth not in methods:
                raise ValueError(
                    f"smooth method must be one of {methods}, got '{smooth}'")
            return smooth, 3
        raise TypeError(
            f"smooth must be None, str, or dict, got {type(smooth)}")
 
 
# ---------------------------------------------------------------------------
# GeostrophicVelocity
# ---------------------------------------------------------------------------
 
class GeostrophicVelocity:
    """
    Geostrophic velocity from SSH, dynamic height anomaly, or both.
 
    The computation mode is determined by which fields are supplied:
 
    ``adt`` only
        Barotropic surface geostrophic velocity.
        u = -(g/f) ∂η/∂y,  v = (g/f) ∂η/∂x
        Result has no pressure dimension.
 
    ``deltaD`` only
        Baroclinic shear velocity without a surface reference.
        u = -(1/f) ∂ΔD/∂y,  v = (1/f) ∂ΔD/∂x
        Result has the same dimensions as deltaD (includes pressure).
 
    ``adt`` + ``deltaD``
        Combined barotropic + baroclinic.
        u_total = u_barotropic + u_baroclinic
        At pressure=0 the barotropic (satellite) values are substituted
        directly because CTD data is typically unreliable at the surface.
 
    Parameters
    ----------
    grid : Grid
        Shared Grid instance from grid.py.  Pass the same instance to
        both GeostrophicVelocity and GradientWindVelocity to maximise
        metric-scale cache reuse.
    adt : xr.DataArray, optional
        Absolute dynamic topography η  (m).
    deltaD : xr.DataArray, optional
        Dynamic height anomaly ΔD  (m² s⁻²).
    lat_name, lon_name : str
        Coordinate names for latitude and longitude.
    smooth : None | str | dict
        Smoothing specification forwarded to StreamFunction.
        Applied once to the first derivatives; curvature and second
        derivatives are computed from the already-smoothed gradients.
    """
 
    def __init__(self, grid, *, adt: xr.DataArray = None,
                 deltaD: xr.DataArray = None,
                 lat_name: str = 'latitude',
                 lon_name: str = 'longitude',
                 smooth=None):
 
        if adt is None and deltaD is None:
            raise ValueError("At least one of 'adt' or 'deltaD' must be provided.")
 
        self.grid     = grid
        self.lat_name = lat_name
        self.lon_name = lon_name
 
        # Build StreamFunction objects for whichever fields were supplied.
        # Both share the same grid so the metric-scale cache is reused.
        self._sf_adt    = StreamFunction(adt,    grid, smooth=smooth) \
                          if adt    is not None else None
        self._sf_deltaD = StreamFunction(deltaD, grid, smooth=smooth) \
                          if deltaD is not None else None
 
        self._result = None   # cached Dataset
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    @property
    def mode(self) -> str:
        """One of 'barotropic', 'baroclinic', or 'combined'."""
        if self._sf_adt is not None and self._sf_deltaD is not None:
            return 'combined'
        if self._sf_adt is not None:
            return 'barotropic'
        return 'baroclinic'
 
    def compute(self, verbose=True) -> xr.Dataset:
        """
        Compute geostrophic velocities.
 
        Parameters
        ----------
        verbose : bool
            If True (default), print the computation mode and relevant
            references on the first call.
 
        Returns
        -------
        xr.Dataset
            Variables: ug, vg, Vg  (m s⁻¹).
            Dimensions match the input field(s).
 
        References
        ----------
        Theory and decomposition:
            Stewart, R.H. (2008). Introduction to Physical Oceanography,
            Chapter 10. Texas A&M University (open access).
            https://geo.libretexts.org/Bookshelves/Oceanography/
            Introduction_to_Physical_Oceanography_(Stewart)
 
            Beal, L.M. Lecture 12: Geostrophic Velocity. University of Miami.
            https://beal-agulhas.earth.miami.edu/teaching/courses/lecture-twelve/

        """
        if self._result is not None:
            return self._result
 
        _REFS = (
            "  Theory : Stewart (2008), Intro to Physical Oceanography, Ch. 10\n"
            "           Beal, L.M., Lecture 12, Univ. of Miami\n"
            "           https://beal-agulhas.earth.miami.edu/teaching/courses/lecture-twelve/"
        )
 
        if self.mode == 'barotropic':
            if verbose:
                print("GeostrophicVelocity [barotropic mode]\n"
                      "  u = -(g/f) dη/dy,  v = (g/f) dη/dx\n"
                      "  Source: SSH/ADT from satellite altimetry\n"
                      + _REFS)
            self._result = self._barotropic()
 
        elif self.mode == 'baroclinic':
            if verbose:
                print("GeostrophicVelocity [baroclinic mode]\n"
                      "  u = -(1/f) dΔD/dy,  v = (1/f) dΔD/dx\n"
                      "  Source: dynamic height anomaly (https://www.teos-10.org/pubs/gsw/html/gsw_geo_strf_dyn_height.html)\n"
                      + _REFS)
            self._result = self._baroclinic()
 
        else:
            if verbose:
                print("GeostrophicVelocity [combined mode]\n"
                      "  u_total = u_baroclinic(ΔD(T,S)) + u_barotropic(SSH)\n"
                      "  u_total = -(1/f) dΔD/dy - (g/f) dη/dy,\n"
                      "  v_total =  (1/f) dΔD/dx + (g/f) dη/dx\n"
                      "  \n"
                      "  surface (p=0)                  : barotropic substituted directly\n"
                      "  below dyn. height ref. level   : masked as NaN\n"
                      "  \n"
                      "  Source:    - dynamic height anomaly (https://www.teos-10.org/pubs/gsw/html/gsw_geo_strf_dyn_height.html)\n"
                      "             - SSH/ADT from satellite altimetry\n"
                      + _REFS)
            self._result = self._combined()
 
        return self._result

 
    # ------------------------------------------------------------------
    # Convenience accessors (needed by GradientWindVelocity)
    # ------------------------------------------------------------------
 
    @property
    def streamfunc(self) -> StreamFunction:
        """
        Primary StreamFunction — adt if available, otherwise deltaD.
        Used by GradientWindVelocity to access curvature and orientation.
        """
        return self._sf_adt if self._sf_adt is not None else self._sf_deltaD
 
    @property
    def fcor(self) -> xr.DataArray:
        """Coriolis parameter broadcast to the primary field's grid."""
        lat = self.streamfunc.field[self.lat_name]
        return _broadcast_fcor(lat, self.streamfunc.field)
 
    @property
    def curvature(self) -> xr.DataArray:
        """Streamline curvature κ (m⁻¹) from the primary stream function."""
        return self.streamfunc.curvature
 
    # ------------------------------------------------------------------
    # Internal computation modes
    # ------------------------------------------------------------------
 
    def _barotropic(self) -> xr.Dataset:
        """u = -(g/f) ∂η/∂y,  v = (g/f) ∂η/∂x"""
        sf   = self._sf_adt
        lat  = sf.field[self.lat_name]
        fcor = _broadcast_fcor(lat, sf.field)
        g    = _broadcast_gravity(lat, sf.field)
 
        ug = -(g / fcor) * sf.dfdy
        vg =  (g / fcor) * sf.dfdx
        return self._package(ug, vg)
 
    def _baroclinic(self) -> xr.Dataset:
        """u = -(1/f) ∂ΔD/∂y,  v = (1/f) ∂ΔD/∂x  (no surface reference)."""
        sf   = self._sf_deltaD
        lat  = sf.field[self.lat_name]
        fcor = _broadcast_fcor(lat, sf.field)
 
        ug = -(1 / fcor) * sf.dfdy
        vg =  (1 / fcor) * sf.dfdx
        return self._package(ug, vg)
 
    def _combined(self) -> xr.Dataset:
        """
        Barotropic + baroclinic shear where CTD coverage exists.
 
        Three zones (in order of priority):
          1. pressure=0 (surface):            barotropic (satellite) always.
          2. CTD-valid levels:                barotropic + baroclinic shear.
          3. Outside CTD / below ref. level:  NaN
        """

        barot_ds = self._barotropic_from(self._sf_adt)
        baroc_ds   = self._baroclinic_from(self._sf_deltaD)

        # broadcast barotropic to the full 3-D grid
        ug_ref = barot_ds['ug'].broadcast_like(baroc_ds['ug'])
        vg_ref = barot_ds['vg'].broadcast_like(baroc_ds['vg'])

        ug_total = ug_ref + baroc_ds['ug']
        vg_total = vg_ref + baroc_ds['vg']

        # combine only where baroclinic is valid; elsewhere NaN
        ctd_valid = np.isfinite(baroc_ds['ug']) & np.isfinite(baroc_ds['vg'])
        ug_total  = xr.where(ctd_valid, ug_total, np.nan)
        vg_total  = xr.where(ctd_valid, vg_total, np.nan)

        # surface level: always use satellite regardless of CTD coverage
        if 'pressure' in self._sf_deltaD.field.dims:
            at_surface = self._sf_deltaD.field['pressure'] == 0
            ug_total   = xr.where(at_surface, ug_ref, ug_total)
            vg_total   = xr.where(at_surface, vg_ref, vg_total)

        return self._package(ug_total, vg_total)
 
    def _barotropic_from(self, sf: StreamFunction) -> xr.Dataset:
        lat  = sf.field[self.lat_name]
        fcor = _broadcast_fcor(lat, sf.field)
        g    = _broadcast_gravity(lat, sf.field)
        ug   = -(g / fcor) * sf.dfdy
        vg   =  (g / fcor) * sf.dfdx
        return self._package(ug, vg)
 
    def _baroclinic_from(self, sf: StreamFunction) -> xr.Dataset:
        lat  = sf.field[self.lat_name]
        fcor = _broadcast_fcor(lat, sf.field)
        ug   = -(1 / fcor) * sf.dfdy
        vg   =  (1 / fcor) * sf.dfdx
        return self._package(ug, vg)
 
    @staticmethod
    def _package(ug: xr.DataArray, vg: xr.DataArray) -> xr.Dataset:
        if isinstance(ug, xr.Dataset):
            ug = ug[list(ug.data_vars)[0]]
        if isinstance(vg, xr.Dataset):
            vg = vg[list(vg.data_vars)[0]]
        Vg = np.sqrt(ug**2 + vg**2)
        return xr.Dataset({'ug': ug, 'vg': vg, 'Vg': Vg})
 
 
# ---------------------------------------------------------------------------
# GradientWindVelocity
# ---------------------------------------------------------------------------
 
class GradientWindVelocity:
    """
    Gradient wind correction on top of geostrophic velocity.
 
    Accepts the same ``adt`` / ``deltaD`` inputs as GeostrophicVelocity
    and constructs it internally, so you never need to instantiate
    GeostrophicVelocity separately unless you want its output too.
 
    Alternatively, pass an already-constructed GeostrophicVelocity via
    the ``geostrophic`` keyword to avoid recomputation.
 
    The gradient wind equation (Holton 2004):
 
        Vgw = -fR/2 ± sqrt(f²R²/4 + fRVg)
 
    Sign classification:
        NH cyclonic     (Rcurv < 0, f > 0):  use minus
        NH anticyclonic (Rcurv > 0, f > 0):  use plus
        SH cyclonic     (Rcurv < 0, f < 0):  use plus
        SH anticyclonic (Rcurv > 0, f < 0):  use minus
 
    Falls back to geostrophic speed where Rcurv = 0 or the discriminant
    is negative (imaginary root — rare, occurs in very tight anticyclones).
 
    Parameters
    ----------
    grid : Grid
        Shared Grid instance.
    adt : xr.DataArray, optional
    deltaD : xr.DataArray, optional
    geostrophic : GeostrophicVelocity, optional
        Pass a pre-computed GeostrophicVelocity to skip its construction.
        If given, adt/deltaD are ignored.
    lat_name, lon_name : str
    smooth : None | str | dict
        Forwarded to GeostrophicVelocity / StreamFunction.
    """
 
    def __init__(self, grid, *, adt: xr.DataArray = None,
                 deltaD: xr.DataArray = None,
                 geostrophic: GeostrophicVelocity = None,
                 lat_name: str = 'latitude',
                 lon_name: str = 'longitude',
                 smooth=None):
 
        if geostrophic is not None:
            self.geo = geostrophic
        elif adt is not None or deltaD is not None:
            self.geo = GeostrophicVelocity(
                grid, adt=adt, deltaD=deltaD,
                lat_name=lat_name, lon_name=lon_name,
                smooth=smooth,
            )
        else:
            raise ValueError(
                "Provide 'adt', 'deltaD', or a pre-built 'geostrophic' instance.")
 
        self._result = None
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def compute(self) -> xr.Dataset:
        """
        Compute gradient wind velocities.
 
        Returns
        -------
        xr.Dataset
            Variables: ugw, vgw, Vgw, uagw, vagw, kappa  (m s⁻¹ / m⁻¹).
        """
        if self._result is not None:
            return self._result
 
        geo_ds = self.geo.compute()
        Vg     = geo_ds['Vg']
        kappa  = self.geo.curvature
        fcor   = self.geo.fcor
        # orient = self.geo.streamfunc.orientation
        orient = np.arctan2(geo_ds['vg'], geo_ds['ug'])
 
        # radius of curvature — masked where κ = 0
        Rcurv = (1.0 / kappa).where(kappa != 0)
 
        # gradient wind discriminant
        discriminant = (fcor**2 * Rcurv**2) / 4 + fcor * Rcurv * Vg
        root         = np.sqrt(discriminant.where(discriminant >= 0))
        half_fR      = -(fcor * Rcurv) / 2
 
        NH      = fcor >= 0
        SH      = fcor <  0
        cyc     = Rcurv <= 0
        anticyc = Rcurv >  0
        fallback = (~np.isfinite(kappa)) | (kappa == 0) | (discriminant < 0)
 
        Vgw = xr.where(fallback,      Vg,
              xr.where(NH & cyc,      half_fR - root,
              xr.where(NH & anticyc,  half_fR + root,
              xr.where(SH & cyc,      half_fR + root,
              xr.where(SH & anticyc,  half_fR - root,
                                      Vg)))))
 
        # project onto Cartesian components using the flow orientation
        # ugw  = -Vgw * np.sin(orient)
        # vgw  =  Vgw * np.cos(orient)
        ugw = Vgw * np.cos(orient)
        vgw = Vgw * np.sin(orient)
        uagw = ugw - geo_ds['ug']
        vagw = vgw - geo_ds['vg']
 
        # Vgw.name = 'Vgw'; ugw.name  = 'ugw'; vgw.name  = 'vgw'
        # uagw.name = 'uagw'; vagw.name = 'vagw'; kappa.name = 'kappa'
 
        self._result = xr.Dataset({
            'ugw': ugw, 'vgw': vgw, 'Vgw': Vgw,
            'uagw': uagw, 'vagw': vagw,
            'kappa': kappa,
        })
        return self._result
 
    @property
    def geostrophic(self) -> xr.Dataset:
        """The underlying geostrophic velocity Dataset."""
        return self.geo.compute()


def streamfunction(
    ds,
    variables=('CT', 'SA', 'p'),
    coordinates=('longitude', 'latitude', 'pressure'),
    p_ref='deepest_common',
):
    """
    p_ref modes
    -----------
    - numeric: single fixed reference pressure for all profiles
    - None: use the deepest pressure common to ALL profiles (the deepest
      level where every profile is still finite).
    - 'deepest_profile': use each profile's OWN deepest pressure as
      its individual reference level. 
    """
    CT = ds[variables[0]]
    SA = ds[variables[1]]
    p_name = variables[2] if variables[2] in ds else coordinates[2]
    if p_name not in ds:
        raise KeyError(
            f"Pressure not found as variable '{variables[2]}' "
            f"or coordinate '{coordinates[2]}'."
        )
    p = ds[p_name]

    # the DIMENSION carrying pressure levels is not always the same string
    # as the pressure VARIABLE name (e.g. variable 'p' along dimension 'pressure')
    p_dim = coordinates[2]
    if p_dim not in SA.dims:
        p_dim = [d for d in SA.dims if d in p.dims][0]

    profile_dim = [d for d in SA.dims if d != p_dim][0]

    finite_mask = np.isfinite(SA) & np.isfinite(CT)

    # Deepest pressure common to all profiles
    all_finite = finite_mask.all(dim=profile_dim)
    deepest_common = ds[p_name].where(all_finite).max()

    if isinstance(p_ref, str) and p_ref == 'deepest_common':
        p_ref = float(deepest_common.values)
        print(f"p_ref = {p_ref} dbar using deepest common pressure")
    elif isinstance(p_ref, str) and p_ref == 'deepest_profile':
        print(f"p_ref takes in each profile the deepest pressure available")
    else:
        p_ref = float(p_ref)
        if p_ref <= deepest_common.values:
            print(f"p_ref = {p_ref} <= deepest common pressure = {deepest_common.values}: \u2713")
        else:
            print(f"Consider setting p_ref to value < {deepest_common.values}")

    if p.ndim == 1:
        p = p.broadcast_like(SA)

    lat = ds[coordinates[1]]
    if 'time' in p.dims:
        g = grav(lat, p.isel(time=0).drop_vars('time'))
    else:
        g = grav(lat, p)
    g.name = 'g'

    axis = SA.get_axis_num(p_dim)

    if isinstance(p_ref, float):

        dim_names = list(SA.dims.keys())
        if ('time' in dim_names) & ('pressure' in dim_names) & ('latitude' in dim_names) & ('longitude' in dim_names):
            SA_chunked = SA.chunk({"time": 1, "pressure": -1, "latitude": "auto", "longitude": "auto"})
            CT_chunked = CT.chunk({"time": 1, "pressure": -1, "latitude": "auto", "longitude": "auto"})
            p_chunked = p.chunk({"time": 1, "pressure": -1, "latitude": "auto", "longitude": "auto"})

            chunk_bytes = np.prod([c[0] for c in SA_chunked.chunks]) * SA.dtype.itemsize
            print(f"Chunk size (~128MB): {chunk_bytes / 1e6:.1f} MB")

            def _persist(*arrays):
                try:
                    client = get_client()
                    return [client.persist(a) for a in arrays]
                except ValueError:
                    return list(arrays)

            SA, CT, p = _persist(SA_chunked, CT_chunked, p_chunked)

        deltaD = xr.apply_ufunc(
            geo_strf_dyn_height, SA, CT, p, p_ref, kwargs={"axis": axis},
            input_core_dims=[["pressure"], ["pressure"], ["pressure"], []],
            output_core_dims=[["pressure"]],
            dask="parallelized",
            output_dtypes=[SA.dtype],
        )
        deltaD = deltaD.compute()

        # single shared p_ref: one vectorized call, as before
        # deltaD = xr.DataArray(
        #     geo_strf_dyn_height(SA, CT, p, p_ref, axis=axis),
        #     dims=SA.dims, coords=SA.coords,
        # )
    else:
        # per-profile p_ref: loop over profiles
        deepest_per_profile = ds[p_name].where(finite_mask).max(dim=p_dim)
        print(f"p_ref = {deepest_per_profile.mean().values:.1f} dbar on average, using deepest pressure of profile")

        deltaD_data = np.full(SA.shape, np.nan)

        for i in range(ds.sizes[profile_dim]):
            sel = {profile_dim: i}
            this_p_ref = deepest_per_profile.isel(**sel).values
            if not np.isfinite(this_p_ref):
                continue  # fully-NaN profile, leave as NaN

            sa_i = SA.isel(**sel)
            ct_i = CT.isel(**sel)
            p_i = p.isel(**sel)
            axis_i = sa_i.get_axis_num(p_dim) if p_dim in sa_i.dims else 0

            dh_i = geo_strf_dyn_height(sa_i, ct_i, p_i, float(this_p_ref), axis=axis_i)

            idx = [slice(None)] * SA.ndim
            idx[SA.dims.index(profile_dim)] = i
            deltaD_data[tuple(idx)] = dh_i

        deltaD = xr.DataArray(deltaD_data, dims=SA.dims, coords=SA.coords)

    deltaD.name = 'deltaD'

    streamfunc = xr.merge([g, deltaD])
    streamfunc.attrs.clear()
    streamfunc = streamfunc.assign_coords(ds.coords).assign_attrs(ds.attrs)

    return streamfunc


# def streamfunction(ds, 
#     variables=('CT', 'SA', 'p'), 
#     coordinates=('longitude', 'latitude', 'pressure'), p_ref=1500):
    
#     CT = ds[variables[0]]
#     SA = ds[variables[1]]
#     p_name = variables[2] if variables[2] in ds else coordinates[2]
#     if p_name not in ds:
#         raise KeyError(
#             f"Pressure not found as variable '{variables[2]}' "
#             f"or coordinate '{coordinates[2]}'.")
#     p = ds[p_name]

#     # Deepest pressure where all stations are still finite
#     try:
#         all_finite = np.isfinite(ds[variables[0]]).all(dim='profile')
#         deepest_common = ds[p_name].where(all_finite).max()
#         print(f"Deepest common pressure: {deepest_common.values} dbar")
#         if p_ref <= deepest_common.values:
#             print(f"p_ref <= {deepest_common.values}: " + u"\u2713")
#         else:
#             print(f'Consider setting p_ref to value < {deepest_common.values}')
#     except ValueError:
#         pass

#     if p.ndim == 1:
#         p = p.broadcast_like(SA)
#     # if lat.ndim == 1:
#     lat = ds[coordinates[1]]#[:, np.newaxis]

#     if 'time' in p.dims:
#         g = grav(lat, p.isel(time=0).drop_vars('time'))
#     else:
#         g = grav(lat, p)
#     g.name = 'g'
    
#     axis = ds[variables[1]].get_axis_num(p_name)
#     # print('%s, axis number: %s' %(p_name, axis))
#     # deltaD = geo_strf_dyn_height(SA, CT, p, p_ref, axis=axis)
#     deltaD = xr.DataArray(
#         geo_strf_dyn_height(SA, CT, p, p_ref, axis=axis),
#         dims=SA.dims, coords=SA.coords,
#         )
#     deltaD.name = 'deltaD'

#     streamfunc = xr.merge([g, deltaD])
#     streamfunc.attrs.clear()
#     streamfunc = streamfunc.assign_coords(ds.coords).assign_attrs(ds.attrs)
#     variables = ['g', 'deltaD']
#     attrs = retrieve_attrs(variables)
#     for var in variables:
#         streamfunc[var] = streamfunc[var].assign_attrs(attrs[var])  
    
#     return streamfunc