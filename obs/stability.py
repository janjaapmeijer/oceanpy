import numpy as np
import math
from scipy.interpolate import interp1d

def mixed_layer_depth(z, ref_depth=20, sig0=False, pt=False, SA=False, CT=False, smooth=False):

    # The reference depth is to avoid large part of the strong diurnal cycle in the top few meters of the ocean.
    # Dong et al. 2008 suggests that 20 m is a sensible determination of "near surface" in the Southern Ocean.
    iref, ref_dep = min(enumerate(z), key=lambda x: abs(abs(x[1]) - ref_depth) and x[1] > ref_depth)

    # smooth profiles with moving average
    if sig0 is not False:
        if smooth:
            N = 5
            sig0 = np.concatenate([np.mean(sig0[:N - 1]) * np.ones(N - 1, ),
                                   np.convolve(sig0.data, np.ones((N,)) / N, mode='valid')])
            sig0 = np.ma.masked_where(sig0 > 1e36, sig0)

        # near-surface value
        sig0_s = sig0[iref]

    if pt is not False:
        if smooth:
            N = 5
            pt = np.concatenate([np.mean(pt[:N - 1]) * np.ones(N - 1, ),
                                 np.convolve(pt.data, np.ones((N,)) / N, mode='valid')])
            pt = np.ma.masked_where(pt > 1e36, pt)

        # near-surface values
        pt_s = pt[iref]

    # Mixed Layer Depth based on de Boyer Montegut et al. 2004's property difference based criteria
    # MLD in potential density difference, fixed threshold criterion (sig0_d - sig0_s) > 0.03 kg/m^3
    if sig0 is not False and SA is False and CT is False:
        imld = iref + next((i for i in range(len(sig0[iref:]))
                            if sig0[i] - sig0_s > 0.03), np.nan)

    # MLD in potential temperature difference, fixed threshold criterion abs(pt_d - pt_s) > 0.2 degC
    if pt is not False:
        imld = iref + next((i for i in range(len(pt[iref:]))
                            if abs(pt[i] - pt_s) > 0.2), np.nan)

    # MLD in potential density and potential temperature difference
    if sig0 is not False and pt is not False:
        imld = iref + next((i for i in range(len(sig0[iref:]))
                            if 0.03 < abs(sig0[i] - sig0_s) < 0.125
                            and 0.2 < abs(pt[i] - pt_s) < 1),
                           next(i for i in range(len(sig0[iref:]))
                                if sig0[i] - sig0_s > 0.03))

    # MLD in potential density with a variable threshold criterion
    if sig0 is not False and SA is not False and CT is not False:
        SA_s = SA[iref]
        CT_s = CT[iref]
        dsig0 = sigma0(SA_s, CT_s - 0.2) - sigma0(SA_s, CT_s)
        imld = iref + next((i for i in range(len(sig0[iref:]))
                            if sig0[i] - sig0_s > dsig0), np.nan)

    return imld, sig0, pt

# def layer_depth(sigma0, z, intervals, ref_depth=20, axis=-1):
#
#     '''
#     Layer depth/ thickness as function of interval.
#     :param sigma0: density
#     :param z: height
#     :param interval: list of density intervals
#     :return: layer depth and layer thickness
#     '''
#
#     if sigma0.shape != z.shape:
#         raise ValueError('The shape of sigma0 and z should match, found %s and %s' % (sigma0.shape, z.shape))
#
#     intervals = np.array(intervals)
#     if intervals.ndim > 1:
#         raise ValueError('The shape of interval should be one-dimensional, found %s' % intervals.ndim)
#     if len(intervals) > max(sigma0.shape):
#         raise ValueError('The length of interval %s should not exceed the size of sigma0[axis], found %s'
#                          % (len(intervals), sigma0.shape[axis]))
#
#     # Number of profiles
#     if sigma0.ndim == 1:
#         sigma0 = sigma0[np.newaxis, :]
#         z = z[np.newaxis, :]
#
#     profiles = sigma0.shape[0]
#
#     depths = np.ones((profiles, len(intervals)))
#     for profile in range(profiles):
#
#         for iv, interval in enumerate(intervals):
#
#             # Check if interval limit is within sigma0 limits
#             if min(sigma0[profile, ]) < interval < max(sigma0[profile, ]):
#
#                 # calculate index of >10 meter reference depth
#                 iref, ref_dep = min(enumerate(z[profile,]), key=lambda x: abs(x[1] - ref_depth) and x[1] > ref_depth)
#
#                 # Search from the minimum error between the observations and the interval limits
#                 # TODO: Ask Helen, closest value or first value that passes threshold, which threshold then?
#                 ic, closest = min(enumerate(sigma0[profile, iref:]), key=lambda x: abs(x[1] - interval))
#                 ic += iref
#
#                 # Interpolate sigma0 over the range [-10 : +10] from the minimum error and
#                 # match the more exact sigma0 with the associated depth(z) value
#                 depth = np.interp(interval, sigma0[profile, ic - 1:ic + 1], z[profile, ic - 1:ic + 1])
#                 depths[profile, iv] = depth
#             #
#             else:
#                 depths[profile, iv] = np.nan
#
#     return depths, abs(np.diff(depths))



def layer_depth(density, z):

    '''
    Calculates the depth of the isopycnals for T-S profiles, from which the density is derived.
    :param density: potential density or neutral density
    :param z: vertical coordinate, height
    :return d: depth of the isopycnals
    '''
    if density.shape != z.shape:
        raise ValueError('The shape of density and z should match, found %s and %s' % (density.shape, z.shape))

    # Number of profiles
    if density.ndim == 1:
        density = density[np.newaxis, :]
        z = z[np.newaxis, :]
    nprofiles = density.shape[0]

    # initialise layer depth variable  #dz
    d = np.ma.masked_all(density.shape)

    # function to determine lower (upper) limits of variable and round down (up)
    def limits(var, ndec=None):
        minvar, maxvar = np.nanmin(var), np.nanmax(var)
        if ndec:
            minvar = math.floor(minvar * 10 ** (ndec)) / (10 ** ndec)
            maxvar = math.ceil(maxvar * 10 ** (ndec)) / (10 ** ndec)
        return minvar, maxvar

    # make a linear spaced array for density, with a small step size
    minvar, maxvar = limits(density, ndec=2)
    step = 0.001
    density_lin = np.linspace(minvar, maxvar, int((maxvar - minvar) / step + 1))

    # bin to make monotonically increasing density
    bins = np.linspace(minvar, maxvar, density.shape[1] + 1)
    ibin = np.digitize(density_lin, bins)
    densitybin = np.array([density_lin[ibin == i].mean() for i in range(1, len(bins))])

    for profile in range(nprofiles):

        # interpolate pressure or depth to linear spaced density
        f = interp1d(density[profile], z[profile], bounds_error=False, fill_value=(np.nan, np.nan))
        zinterp = f(density_lin)

        if np.isfinite(zinterp[-1]):
            idx = np.where(zinterp == zinterp[-1])[0]
            if len(idx) > 1:
                zinterp[idx[1:]] = np.nan

        # average interpolated values over bin
        zbin = [zinterp[ibin == i].mean() for i in range(1, len(bins))]

        # # determine dz between depths
        # dz[profile] = abs(np.gradient(zbin))

        # store profile depths in array
        d[profile] = zbin

    return densitybin, d