import sys, os

import numpy as np
from scipy.interpolate import UnivariateSpline

from netCDF4 import Dataset, default_fillvals
from .tools.netcdf import createNetCDF

from shutil import copyfile

if sys.version_info[0] == 3:
    from gsw import SA_from_SP, CT_from_t, pt_from_t, sigma0, spiciness0, \
        z_from_p, grav, geo_strf_dyn_height, geostrophic_velocity
if sys.version_info[0] == 2:
    from pygamman import gamman as nds

varis = {
    'SA': ('sea_water_absolute_salinity', 'f8'),
    'CT': ('sea_water_conservative_temperature', 'f8'),
    'g': ('gravitational_acceleration', 'f8'),
    'z': ('height', 'f8'),
    'pt': ('sea_water_potential_temperature', 'f8'),
    'sigma0': ('sea_water_sigmat', 'f8'),
    'spiciness0': ('sea_water_spiciness', 'f8'),
    'deltaD': ('dynamic_height_anomaly', 'f8'),
    'gamman': ('sea_water_neutral_density', 'f8')
}

class CTDprofiles(object):


    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
#         self.teos10_vars = teos10_vars

        self.dsin = Dataset(input_file, 'r+')
        if os.path.isfile(output_file):
            print('Output file %s already exists.' %os.path.basename(output_file))
            self.dsout = createNetCDF(output_file)
        else:
            copyfile(input_file, output_file)
            print('Output file %s, copied from input file %s.'
                  %(os.path.basename(output_file), os.path.basename(input_file)))
            self.dsout = createNetCDF(output_file)

    def vars_exist(self, file, variables, group=None):
        filename = os.path.basename(file)

        if group in self.dsout.dataset.groups:
            exist = [var in self.dsout.dataset[group].variables for var in variables]
            na = [var for var in variables if var not in self.dsout.dataset[group].variables]
        else:
            print('Group %s, does not exist, checking for variables in root ...' %group)
            exist = [var in self.dsout.dataset.variables for var in variables]
            na = [var for var in variables if var not in self.dsout.dataset.variables]

        if all(exist):
            raise Exception('All variables %s already exist in output file:\n%s.'
                  %(', '.join(variables[exist]), filename))
        elif na == ['gamman']:
            if sys.version_info[0] == 3 and group == 'TEOS10':
                raise Exception('All variables except for %s exist, run function calculate_teos10 in Python 2\n' %', '.join(na),
                      'to store variable in output_file: %s.' %filename)
            elif sys.version_info[0] == 3 and group == 'GEM':
                raise Exception('All variables except for %s exist, run function create_gem in Python 2\n' %', '.join(na),
                      'to store variable in output_file: %s.' %filename)
            else:
                print('Variable %s does not exist, switch to Python 2 environment' %tuple(na),
                      'to calculate and store variable in output_file: %s' %filename)
        else:
            if group == 'TEOS10':
                print('Variables %s do not exist, run calculate_teos10 function \n' %', '.join(na),
                      'to calculate and store variables in output_file: %s' %filename)
            elif group == 'GEM':
                print('Variables %s do not exist, run create_gem function \n' %', '.join(na),
                      'to calculate and store variables in output_file: %s' %filename)
            else:
                print('Variables %s do not exist.' %', '.join(na))

        return (na, filename)

    def calculate_teos10(self, variables, p_ref=None, group='TEOS10',
                         coordinates=('pressure', 'latitude', 'longitude'),
                         dimensions=('profile', 'pressure')):

        na, filename = self.vars_exist(self.output_file, variables=variables, group=group)

        try:
            p = self.dsin[coordinates[0]][:]
            lat = self.dsin[coordinates[1]][:][:, np.newaxis]
            lon = self.dsin[coordinates[2]][:][:, np.newaxis]
        except IndexError as error:
            raise('Coordinates: %s are not availble in input file.' %list(coordinates))

        try:
            t = self.dsin['t'][:]
            SP = self.dsin['SP'][:]
        except IndexError as error:
            raise('In-situ temperature (t) and Practical Salinity ($S_p$) are necessary\n',
                  'for the calculation of: %s.' %na)

        if sys.version_info[0] == 3:

            SA = SA_from_SP(SP, p, lon, lat)
            CT = CT_from_t(SA, t, p)

            # store calculated variables in variables dictionary
            new_variables = {}
            for var in na:
                if var not in new_variables.keys():
                    if var is 'SA':
                        new_variables['/%s/%s' %(group, var)] = varis[var] + (dimensions, ) + (SA,)
                    elif var is 'CT':
                        new_variables['/%s/%s' %(group, var)] = varis[var] + (dimensions, ) + (CT,)
                    elif var is 'pt':
                        if p_ref is None:
                            p_ref = 0
                        new_variables['/%s/%s' %(group, var)] = varis[var] + (dimensions, ) + (pt_from_t(SA, t, p, p_ref),)
                    elif var is 'deltaD':
                        if p_ref is None:
                            p_ref = 1500
                            raise Warning('Reference pressure is assumed to be 1500 dbar.')
                        new_variables['/%s/%s' %(group, var)] = varis[var] + (dimensions, ) + (np.ma.masked_invalid(geo_strf_dyn_height(SA.data, CT.data, p, p_ref=p_ref, axis=1)),)
                    elif var is 'g':
                        new_variables['/%s/%s' %(group, var)] = varis[var] + (dimensions, ) + (grav(lat, p),)
                    elif var is 'z':
                        new_variables['/%s/%s' %(group, var)] = varis[var] + (dimensions, ) + (z_from_p(p, lat),)
                    elif var is 'g':
                        new_variables['/%s/%s' %(group, var)] = varis[var] + (dimensions, ) + (grav(lat, p),)
                    elif var is 'sigma0':
                        new_variables['/%s/%s' %(group, var)] = varis[var] + (dimensions, ) + (sigma0(SA, CT),)
                    elif var is 'spiciness0':
                        new_variables['/%s/%s' %(group, var)] = varis[var] + (dimensions, ) + (spiciness0(SA, CT),)
                    elif var is 'gamman':
                        print('Switch to python 2 environment to calculate: %s.' %var)
                    else:
                        print('Currently, not supporting function to calculate: %s.' %var)

            # save data in netcdf file using OceanPy's createNetCDF class
            self.dsout.create_vars(new_variables)
            print('New variables %s, stored in group %s, of output file.'
                  % (', '.join([var for var in na if var in self.dsout.dataset[group].variables.keys()]),
                     group))

        if sys.version_info[0] == 2:

            if 'gamman' not in self.dsout.dataset.variables.keys():
                gamman = np.ma.masked_all((len(self.dsin.dimensions[dimensions[0]]), len(self.dsin.dimensions[dimensions[1]])))
                for i in range(0, len(self.dsin.dimensions[dimensions[0]])):
                    try:
                        t = np.ma.masked_invalid(t)
                        t.data[t.mask] = default_fillvals['f8']
                        SP = np.ma.masked_invalid(SP)
                        SP.data[SP.mask] = default_fillvals['f8']

                        gn = nds.gamma_n(SP[i,].data, t[i,].data, p, p.size, lon[i, 0], lat[i, 0])[0]
                        mask = SP[i,].mask | t[i,].mask
                        gn[mask] = np.nan

                    except AttributeError:
                        gn = np.zeros(len(self.dsin.dimensions[dimensions[1]]))
                        gn[:] = np.nan
                    except Exception as e:
                        gn = nds.gamma_n(SP[i,], t[i,], p, p.size, lon[i, 0], lat[i, 0])[0]
                    gamman[i,] = gn

                new_variable = {'/%s/gamman' %group: varis['gamman'] + (dimensions, ) + (gamman, )}

                # save data in netcdf file using OceanPy's createNetCDF class
                gem = self.dsout.dataset.createGroup(group)
                for name, dimension in self.dsin.dimensions.items():
                    gem.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
                self.dsout.create_vars(new_variable)

                print('New variable %s, stored in group %s, of the output file.'
                      % ('gamman' if 'gamman' in self.dsout.dataset[group].variables else None, group))

    def create_gem(self, outliers=None, p_int=2, minobs=25, group='GEM',
                   variables=('CT', 'SA', 'gamman'),
                   coordinates=('pressure', 'latitude', 'longitude'),
                   dimensions=('profile', 'pressure')):

        na, filename = self.vars_exist(self.output_file, variables=variables, group=group)

        p = self.dsin[coordinates[0]][:]
        stations = self.dsin['station']
        deltaD = self.dsin['TEOS10']['deltaD']
        g = self.dsin['TEOS10']['g']

        # TODO: check if TEOS10 variables exist and stored in netcdf group: 'TEOS10'

        istations = [istat for (istat, station) in enumerate(stations) if station not in outliers]

        # find indices for pressure levels in CTD prodiles
        pressure_levels = {}
        for ip, pres in enumerate(p):
            pressure_levels[pres] = ip

        # get dynamic height contours
        D = np.array([deltaD[profile, pressure_levels[p_int]] / g[profile, pressure_levels[p_int]]
              for profile in range(self.dsin.dimensions[dimensions[0]].size)])

        # create cubic spline fit at each pressure level
        splines = {}
        for pres in p:
            splines[pres] = {}
            for var in variables:
                mask = np.ma.masked_where((np.isnan(D)) |
                                          (np.ma.masked_invalid(self.dsin['TEOS10'][var][:, pressure_levels[pres]]).mask) |
                                          ([station in outliers for station in stations]), D).mask
                try:
                    x, y = zip(*sorted(zip(D[~mask], self.dsin['TEOS10'][var][:, pressure_levels[pres]][~mask])))
                    if len(x) >= minobs:
                        splines[pres][var] = UnivariateSpline(x, y)
                    else:
                        splines[pres][var] = np.isnan(x)
                except:
                    splines[pres][var] = []

        # create GEM
        new_variables = {}
        for var in variables:
            new_variable = np.ma.masked_all((self.dsin.dimensions[dimensions[0]].size,
                                         self.dsin.dimensions[dimensions[1]].size))
            for ip, pres in enumerate(p):
                try:
                    for id, dynh in enumerate(D):
                        new_variable[id, ip] = float(splines[pres][var](dynh))
                except TypeError:
                    pass
            new_variables['/%s/%s' %(group, var)] = varis[var] + (dimensions, ) + (new_variable,)

        # save data in netcdf file using OceanPy's createNetCDF class
        gem = self.dsout.dataset.createGroup(group)
        for name, dimension in self.dsin.dimensions.items():
            gem.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
        self.dsout.create_vars(new_variables)

        print('New variables %s, stored in group %s, of the output file.'
              % (', '.join([var for var in na if var in self.dsout.dataset[group].variables.keys()]), group))

    def close(self):
        self.dsin.close()
        self.dsout.dataset.close()
