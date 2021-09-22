def wind_profile_power_law(u_ref, z_ref, z=10, alpha=1/7):
    '''
    wind_profile_power_law returns the wind speed at a requested height,
    relative to the known wind speed at a reference height
    '''
    u = u_ref * (z / z_ref)**alpha
    return u