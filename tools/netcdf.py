import os
from netCDF4 import Dataset, date2num
import xml.etree.ElementTree as ElementTree
from datetime import datetime
import numpy as np
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

__all__ = ['createNetCDF']

class createNetCDF(object):

    def __init__(self, output_file):
        self.output_file = output_file
        if os.path.isfile(output_file):
            self.dataset = Dataset(output_file, 'r+')
        else:
            self.dataset = Dataset(output_file, 'w')

    def add_dims(self, dims):
        ''' Create dimensions of variables to store in NetCDF '''
        for dimname, size in dims.items():
            self.dataset.createDimension(dimname=dimname, size=size)
        # TODO: let function work inside netcdf

    def add_glob_attr(self, glob_attr):
        ''' Add global attributes '''
        time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        attr = {'Conventions': 'CF-1.6',
                'Metadata_Conventions': 'Unidata Dataset Discovery v1.0',
                'history': 'Created on %s with %s' % (time_now, os.path.basename(__file__))
                }
        for key in attr.keys():
            if not hasattr(self.dataset, 'Conventions') and key not in glob_attr.keys():
                setattr(self.dataset, key, attr[key])

        for key in glob_attr.keys():
            setattr(self.dataset, key, glob_attr[key])

    def create_vars(self, vars):

        # open cf conventions standard names xml file
        root = ElementTree.parse(
            urlopen('http://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml'))

        # check if var_name is in standard names xml file and if so save in dict with according unit and description
        standard_names = {}
        for child in root.findall('entry'):
            standard_names[child.get('id')] = (child.findtext('canonical_units'), child.findtext('description'))

        for varname, values in vars.items():
            standard_name, datatype, dimensions, data = values

            # if '/' in varname:
            #     group = os.path.dirname(varname)
            #     self.dataset.createGroup(group)

            # create variables
            var = self.dataset.createVariable(varname, datatype, dimensions)

            # add variable attributes
            if standard_name.split()[-1] in standard_names.keys():
                var.standard_name = standard_name.split()[-1]
                if standard_name.split()[-1] == 'time':
                    calendar = 'standard'
                    var.units = 'seconds since 1970-01-01 00:00'
                    data = date2num(data, units=var.units, calendar=calendar)
                else:
                    var.units = standard_names[standard_name.split()[-1]][0]
            var.long_name = standard_name
            # var.description = standard_names[varname.split()[-1]][1]

            # add data to variables
            var[:] = np.ma.filled(data.astype(float), np.nan) # data

    def delete_vars(self, vars):
        # https://stackoverflow.com/questions/15141563/python-netcdf-making-a-copy-of-all-variables-and-attributes-but-one/32002401
        pass

    def close(self):
        self.dataset.close()
