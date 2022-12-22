import numpy as np
import xarray as xr
from skimage.measure import find_contours
from . import haversine, rotatexy, cartesian_to_natural, bearing_to_standard
import pyproj

__all__ = ['Contour']

class Contour(object):

    def __init__(self, filename_obj, coords=None, contour_name=None, contour_value=None):
        try:
            self.dataset = xr.open_dataset(filename_obj)
        except ValueError:
            self.dataset = filename_obj
        if coords is None:
            self.coords = list(self.dataset.coords.keys())
            # if len(coords) == 2:
            #     self.coords_contour = coords
            # else:
            #     raise ValueError(
            #         'Specify which coordinates from %s are the contour coords, with parameter `coords`'
            #         %coords)
        else:
            self.coords = coords
        # self.length = self.contour_length(contour_name, contour_value)

        # self.cont_coords = self.get_coords(contour_name, contour_value)
        #     pass

    def get_coords(self, contour_name, contour_value,
                   section_var=None, interp=False, spacing=None):

        """
        Coordinates of contour with contour_value.

        Parameters
        ----------
        contour_name : str
            Name of variables in Dataset.
        contour_value : float
            Value of contour to select.

        """

        # if not all([coord in self.coords for coord in self.dataset[contour_name].coords]):
        #     raise ValueError('Contour field dimensions %s do not correspond with coordinates %s',
        #                      %(self.dataset[contour_name].coords, self.coords))

        # find longest contour
        contours = find_contours(self.dataset[contour_name].values, contour_value)
        contour = max(contours, key=lambda x: len(x))

        # contour indexes
        idx_x, idx_y = contour[:, 1], contour[:, 0]
        ix_floor, iy_floor = np.floor(idx_x).astype(int), np.floor(idx_y).astype(int)
        ix_ceil, iy_ceil = np.ceil(idx_x).astype(int), np.ceil(idx_y).astype(int)

        # contour coordinates and distance
        var = self.dataset[contour_name] if section_var is None else self.dataset[section_var]

        x_cont = var.isel({self.coords[0]: ix_floor})[self.coords[0]]
        + (idx_x - ix_floor) * (var.isel({self.coords[0]: ix_ceil})[self.coords[0]].values
           - var.isel({self.coords[0]: ix_floor})[self.coords[0]].values)
        y_cont = var.isel({self.coords[1]: iy_floor})[self.coords[1]]
        + (idx_y - iy_floor) * (var.isel({self.coords[1]: iy_ceil})[self.coords[1]].values
           - var.isel({self.coords[1]: iy_floor})[self.coords[1]].values)

        # remove duplicate coordinates
        cont_coords = np.array(list(zip(x_cont[self.coords[0]].values,
                                        y_cont[self.coords[1]].values)))
        if interp:
            cont_coords = self.contour_interp(cont_coords, spacing)
        self.cont_coords = np.unique(cont_coords, axis=0)

        # return cont_coords

    # interp contour with specific distance between contour points
    def contour_interp(self, cont_coords, spacing):

        """
        Parameters
        ----------
        cont_coords :
        spacing : float
            Distance between contour points in meter.
        """

        cumdist = np.cumsum(haversine(cont_coords[:, 0], cont_coords[:,1])[0])
        cont_dist = np.hstack([[0], cumdist])
        cont_dist_interp = np.linspace(cont_dist[0], cont_dist[-1],
                                       int(cont_dist[-1] / spacing) + 1)

        x_cont = np.interp(cont_dist_interp, cont_dist, cont_coords[:, 0])
        y_cont = np.interp(cont_dist_interp, cont_dist, cont_coords[:, 1])

        cont_coords=list(zip(x_cont, y_cont))

        return cont_coords

    def make_coordinate(self, da, coord_name=None, coords=None):

        if coords is None:
            coords = self.coords

        # to make a coordinate 2 dimensions are needed,
        # adjusts shape of coordinate in case the size =1
        coord_sizes = np.array([da[coord].size for coord in coords])
        if any(coord_sizes == 1) and coord_sizes.size == 2:
            shp, idx = np.max(coord_sizes), np.argmin(coord_sizes)
            da = da.assign({coords[idx]: np.broadcast_to(da[coords[idx]], shp)})
            if not coord_sizes.size == 2:
                raise ValueError(
                    'To make a 1 dimension sectional coordinate 2 coordinates, should be given, but found %s'
                    %coord_sizes.size)

        # make contour coordinate along selected contour points
        # try:
        distance = haversine(da[coords[0]], da[coords[1]])[0]
        # except:
            # pass

        cumdist = np.cumsum(distance)
        cont_coord = np.hstack([[0], cumdist])

        return da.assign_coords({coord_name: cont_coord})

    def contour_length(self, contour_name, contour_value):

        """
        Calculate length of specific contour.
        """

        try:
            self.cont_coords
        except:
            self.get_coords(contour_name, contour_value)

        return np.sum(haversine(self.cont_coords[:, 0], self.cont_coords[:,1])[0])

    def cross_section(self, coord_sel, npnts, spacing,
                      section_name='cross-section', var_name=None, coords=None, transform=None, decompose=False):

        """
        Vertical section across contour

        Parameters
        ----------

        """

        try:
            self.cont_coords
        except:
            self.get_coords()

        if coords is None:
            coords = self.coords

        # find closest contour point and find bearing with respect to contour line
        idx = np.argmin([haversine([ln, coord_sel[0]], [lt, coord_sel[1]])[0]
                         for ln, lt in self.cont_coords])
        bearing = haversine(*zip(*self.cont_coords))[1]
        orientation = bearing_to_standard(sum(bearing[idx-1:idx+1])/2)
        orientation = orientation - (np.pi/2) if (np.pi/2) < orientation <= (3*np.pi/2) else orientation + (np.pi/2)

        # transform center point of transect to cartesian coordinate
        WGS84 = pyproj.Proj('EPSG:4326')
        xc_ts, yc_ts = pyproj.transform(WGS84, transform, *np.flip(self.cont_coords[idx]))

        # calculate remaining stations of transect
        xi_st = np.linspace(xc_ts - (npnts - 1) * spacing / 2,
                            xc_ts + (npnts - 1) * spacing / 2, npnts)
        yi_st = yc_ts * np.ones(npnts)

        # rotate stations with orientation angle from x-axis
        x_st, y_st = zip(*rotatexy(xc_ts, yc_ts, xi_st, yi_st, orientation))

        # transform station coordinates back to latlon
        lat_st, lon_st = pyproj.transform(transform, WGS84, x_st, y_st)

        # TODO; temporary fix for negative longitudes, especially in models
        lat_st = np.array(lat_st)
        if all(self.dataset[coords[0]] < 0):
            lon_st = np.array(lon_st) - 360

        cross_cont_pnt = []
        for x, y in zip(lon_st, lat_st):
            cross_cont_pnt.append(
                self.dataset.sel({coords[0]: x, coords[1]: y},
                                           method='nearest'))
        ds = xr.concat(cross_cont_pnt, dim=section_name)
        ds = self.make_coordinate(ds, section_name, coords)

        if decompose and type(var_name) == tuple:
            ds = self.decompose_vector(ds, var_name, coords, cross=True)

        return ds



    def along_section(self, var_name=None, section_name='section', coords=None, decompose=False):

        """
        Vertical section along a contour.

        Parameters
        ----------
        ds -- xarray dataset
        contour_value -- the value for the streamline or contour to follow
        variable -- variable in dataset to get along contour section
        timemean -- take time mean of the dataset (default True)

        Returns
        -------

        Examples
        --------

        """

        try:
            self.cont_coords
        except:
            self.get_coords()

        if coords is None:
            coords = self.coords

        # select variable at contour points along contour section
        # var = contour_var if section_var is None else section_var
        # coords = xyz_coords if section_var is None else section_coords
        var_cont_pnt = []

        for x, y in self.cont_coords:
            var_cont_pnt.append(
                self.dataset.sel({coords[0]: x, coords[1]: y},
                                           method='nearest'))
        da = xr.concat(var_cont_pnt, dim=section_name)

        # make new coordinate along contour
        da = self.make_coordinate(da, section_name, coords)
        # self.dataset[var_name+'-'+section_name] = da

        if decompose and type(var_name) == tuple:
            da = self.decompose_vector(da, var_name, coords)

        return da

    def decompose_vector(self, da, var_names, coords, cross=False):

        """ """

        bearing = haversine(da[coords[0]], da[coords[1]])[1]
        bearing = np.concatenate((bearing, bearing[-1:]))
        if cross:
            bearing = bearing + 90
        # bearing[1:] = (bearing[1:] + bearing[:-1]) / 2 # central difference
        bearing = np.broadcast_to(bearing, da[var_names[0]].T.shape).T



        # decompose u,v-velocities along contour
        # print('u : ', da[var_names[0]])
        ut, vn = cartesian_to_natural(
            da[var_names[0]], da[var_names[1]], bearing, bearing=True)
        # print('ut : ', ut)
        da = da.assign({(var_names[0]+'t') : ut, (var_names[1]+'n') : vn})

        return da
