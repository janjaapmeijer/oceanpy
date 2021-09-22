# https://ocefpaf.github.io/python4oceanographers/blog/2015/06/22/osm/

import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib.pyplot as plt

def make_map(projection=ccrs.PlateCarree(), figsize=None):
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=projection))
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    return fig, ax, gl

def make_map_subplt(projection=ccrs.PlateCarree()):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), sharey=True,
                           subplot_kw=dict(projection=projection))
    gl = ax1.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl2 = ax2.gridlines(draw_labels=True)
    gl2.xlabels_top = gl2.ylabels_left = gl2.ylabels_right = False
    gl2.xformatter = LONGITUDE_FORMATTER

    return fig, (ax1, ax2)

# # GOOGLE MAPS
# import cartopy.io.img_tiles as cimgt
#
# extent = [-39, -38.25, -13.25, -12.5]
#
# request = cimgt.GoogleTiles()
#
# fig, ax = make_map(projection=request.crs)
# ax.set_extent(extent)
#
# ax.add_image(request, 10)
#
#
# # OPEN STREET MAPS
# extent = [14.1, 14.55, 55.75, 56.05]
# # activate oceanv3
# # -clipsrc x_min y_min x_max y_max
# # ogr2ogr -f "ESRI Shapefile" <output>.shp <input>.shp -clipsrc -180 0 180 90
# # ogr2ogr -f "ESRI Shapefile" output.shp lines.shp -clipsrc 14 55 15 57
#
# # coastline + coloured land
# fig, ax = make_map(projection=ccrs.PlateCarree())
# ax.set_extent(extent)
#
# shp = shapereader.Reader('c:/Users/jaap.meijer/Downloads/land-polygons-complete-4326/output')
# for record, geometry in zip(shp.records(), shp.geometries()):
#     ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='lightgray',
#                       edgecolor='black')
#
# # only coastline
# fig, ax = make_map(projection=ccrs.PlateCarree())
# ax.set_extent(extent)
#
# shp = shapereader.Reader('c:/Users/jaap.meijer/Downloads/coastlines-split-4326/output')
# for record, geometry in zip(shp.records(), shp.geometries()):
#     ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='w',
#                       edgecolor='black')
