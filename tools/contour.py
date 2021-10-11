import numpy as np
import xarray as xr
from skimage.measure import find_contours
from . import haversine, rotatexy
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
        else:
            self.coords = coords

        # self.get_coords(contour_name=contour_name, contour_value=contour_value)
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

        if not all([coord in self.coords for coord in self.dataset[contour_name].coords]):
            raise ValueError('Contour field dimensions %s do not correspond with coordinates %s',
                             (self.dataset[contour_name].coords, self.coords))

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

    def make_coordinate(self, da, coord_name=None):

        # make contour coordinate along selected contour points
        # try:
        distance = haversine(da[self.coords[0]], da[self.coords[1]])[0]
        # except:
            # pass

        cumdist = np.cumsum(distance)
        cont_coord = np.hstack([[0], cumdist])

        return da.assign_coords({coord_name: cont_coord})

    def length(self, contour_value):

        """
        Calculate length of specific contour with contour_value.
        """

        try:
            self.cont_coords
        except:
            self.get_coords()

        self.length = np.sum(haversine(cont_coords[:, 0], cont_coords[:,1])[0])

    def cross_section(self, coord_sel, npnts, spacing,
                      section_name='cross-section', coords=None, transform=None):

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
        # orientation = self.bearing2standard(bearing[idx])
        orientation = self.bearing2standard(bearing[idx-1:idx])/2
        orientation = orientation - (np.pi/2)
        print(orientation)
        # transform center point of transect to cartesian coordinate
        WGS84 = pyproj.Proj('EPSG:4326')
        xc_ts, yc_ts = pyproj.transform(WGS84, transform, *np.flip(self.cont_coords[idx]))
        print(xc_ts, yc_ts)
        # calculate remaining stations of transect
        xi_st = np.linspace(xc_ts - (npnts - 1) * spacing / 2,
                            xc_ts + (npnts - 1) * spacing / 2, npnts)
        yi_st = yc_ts * np.ones(npnts)

        # rotate stations with orientation angle from x-axis
        x_st, y_st = zip(*rotatexy(xc_ts, yc_ts, xi_st, yi_st, orientation))

        # transform station coordinates back to latlon
        lat_st, lon_st = pyproj.transform(transform, WGS84, x_st, y_st)

        cross_cont_pnt = []
        for x, y in zip(lon_st, lat_st):
            cross_cont_pnt.append(
                self.dataset.sel({coords[0]: x, coords[1]: y},
                                           method='nearest'))
        ds = xr.concat(cross_cont_pnt, dim=section_name)
        # ds = self.make_coordinate(ds, section_name)

        return lon_st, lat_st, ds



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
        da = self.make_coordinate(da, section_name)
        # self.dataset[var_name+'-'+section_name] = da

        if decompose and type(var_name) == tuple:
            da = self.decompose(da, var_name)

        return da

    def decompose(self, da, var_names):

        """ """

        bearing = haversine(da[self.coords[0]], da[self.coords[1]])[1]

        # decompose u,v-velocities along contour
        theta = self.bearing2standard(bearing)
        theta = np.broadcast_to(theta, da[var_names[0]][:-1].T.shape).T

        da[var_names[0]+'_t'] = da[var_names[0]][:-1] * np.cos(theta)
        + da[var_names[1]][:-1] * np.sin(theta)
        da[var_names[1]+'_n'] = da[var_names[0]][:-1] * np.sin(theta)
        - da[var_names[1]][:-1] * np.sin(theta)

        return da

    def bearing2standard(self, bearing):
        return np.deg2rad((90 - bearing) % 360)
