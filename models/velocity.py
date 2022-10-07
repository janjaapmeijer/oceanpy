import numpy as np
import xarray as xr
from gsw import f

__all__ = ['geostrophic_velocity', 'ekman_velocity']

def geostrophic_velocity(ds, grid, sea_level='sea_level', stream_func='deltaD', gravity='gu', coriolis='f', delta_names=('dx', 'dy')):

    '''
    calculate geostrophic velocity from level of 'known' motion (e.g. surface)
    '''

    # surface geostrophic velocity
    detadx = grid.interp(grid.diff(ds[sea_level], 'X', boundary='extend'), 'Y', boundary='extend') / ds[delta_names[0]]
    detady = grid.interp(grid.diff(ds[sea_level], 'Y', boundary='extend'), 'X', boundary='extend') / ds[delta_names[1]]

    ds['ug_s']= - (ds[gravity].isel(st_ocean=0) / ds[coriolis]) * detady
    ds['vg_s'] = (ds[gravity].isel(st_ocean=0) / ds[coriolis]) * detadx

    ds['ug_s'].name = 'ug_s'
    ds['ug_s'].attrs['standard_name'] = 'surface_geostrophic_eastward_sea_water_velocity'
    ds['ug_s'].attrs['long_name'] = r'$u_g,s$'
    ds['ug_s'].attrs['units'] = r'$\mathrm{ms}^{-1}$'

    ds['vg_s'].name = 'vg_s'
    ds['vg_s'].attrs['standard_name'] = 'surface_geostrophic_northward_sea_water_velocity'
    ds['vg_s'].attrs['long_name'] = r'$v_g,s$'
    ds['vg_s'].attrs['units'] = r'$\mathrm{ms}^{-1}$'

    ds['ug'] = ds['ug_s'] + grid.interp(grid.diff(ds[stream_func], 'Y', boundary='extend'), 'X', boundary='extend')  / (ds[delta_names[1]] * ds[coriolis])
    ds['vg'] = ds['vg_s'] - grid.interp(grid.diff(ds[stream_func], 'X', boundary='extend'), 'Y', boundary='extend')  / (ds[delta_names[0]] * ds[coriolis])

    ds['ug'].name = 'ug'
    ds['ug'].attrs['standard_name'] = 'geostrophic_eastward_sea_water_velocity'
    ds['ug'].attrs['long_name'] = r'$u_g$'
    ds['ug'].attrs['units'] = r'$\mathrm{ms}^{-1}$'

    ds['vg'].name = 'vg'
    ds['vg'].attrs['standard_name'] = 'geostrophic_northward_sea_water_velocity'
    ds['vg'].attrs['long_name'] = r'$v_g$'
    ds['vg'].attrs['units'] = r'$\mathrm{ms}^{-1}$'

    return ds

def ekman_velocity(ds, grid, rho_0=1025, A_z=8e-3, hemisphere='N'):

    # Coriolis parameter
    fcor,_ = xr.broadcast(f(ds.yu_ocean), ds.xu_ocean)

    # u_e = (1 / (rho_0 * (2 * A_z * np.abs(fcor))**(1/2))) * (ds.tau_x - ds.tau_y)
    # v_e = (1 / (rho_0 * (2 * A_z * np.abs(fcor))**(1/2))) * (ds.tau_x + ds.tau_y)
    # ds['tau'] = (ds.tau_x**2 + ds.tau_y**2)**(1/2)
    V_x0 = ds.tau_x / (rho_0**2 * np.abs(fcor) * A_z)**(1/2)
    V_y0 = ds.tau_y / (rho_0**2 * np.abs(fcor) * A_z)**(1/2)
    a = (np.abs(fcor) / (2 * A_z))**(1/2)

    phase = np.pi / 4

    ds['ue'] = V_x0 * np.exp(a * -ds.st_ocean) * np.cos(-phase + (a * -ds.st_ocean)) + V_y0 * np.exp(a * -ds.st_ocean) * np.cos(phase + (a * -ds.st_ocean)) #
    ds['ve'] = V_x0 * np.exp(a * -ds.st_ocean) * np.sin(-phase + (a * -ds.st_ocean)) + V_y0 * np.exp(a * -ds.st_ocean) * np.sin(phase + (a * -ds.st_ocean)) #

    if hemisphere == 'S':
        ds['ue'] = V_x0 * np.exp(a * -ds.st_ocean) * np.cos(-phase + (a * -ds.st_ocean)) - V_y0 * np.exp(a * -ds.st_ocean) * np.cos(phase + (a * -ds.st_ocean))
        ds['ve'] = - V_x0 * np.exp(a * -ds.st_ocean) * np.sin(-phase + (a * -ds.st_ocean)) + V_y0 * np.exp(a * -ds.st_ocean) * np.sin(phase + (a * -ds.st_ocean))

    return ds
