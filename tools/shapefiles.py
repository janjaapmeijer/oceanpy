import OceanPy as op
import pyproj

import shapefile
import shapely.geometry as slgeo
import matplotlib.patches as mplpatch

class Shapefile(object):

    shpfile = ''

    def __init__(self, shpfile=None, projin=None, projout=None, pathwrite=None, zaxis=False):
        self.shpfile = shpfile
        self.projin = projin
        self.projout = projout
        # self.pathwrite = pathwrite
        self.get_coords(zaxis=zaxis)
        # self.get_patch()

    def read_shpfile(self):
        try:
            self.reader = shapefile.Reader(self.shpfile)
        except shapefile.ShapefileException:
            self.reader = self.shpfile

    def get_coords(self, zaxis):
        self.read_shpfile()
        self.coordinates = []
        for shape in self.reader.shapes():
            if all([len(x) > 2 for x in shape.points]) and zaxis:
                for point in shape.points:
                    if self.projin is not None and self.projout is not None:
                        self.coordinates.append(pyproj.transform(self.projin, self.projout, point[0], point[1]) + (point[2],))
                    else:
                        self.coordinates.append((point[0], point[1], point[2]))

            else:
                for point in shape.points:
                    if self.projin is not None and self.projout is not None:
                        self.coordinates.append(pyproj.transform(self.projin, self.projout, point[0], point[1]))
                    else:
                        self.coordinates.append((point[0], point[1]))

    def transform_shapes(self):
        if self.reader.shapeType == 1:
            pass

        elif self.reader.shapeType == 3:
            self.writer = shapefile.Writer(shapeType=3)
            for shape in self.reader.shapes():
                for point in shape.points:
                    x, y = pyproj.transform(self.projin, self.projout, point[0], point[1])
                    self.writer.point(x, y)
        elif self.reader.shapeType == 5:
            self.writer = shapefile.Writer(shapefile.POLYGON)
            for shape in self.reader.shapes():
                points = shape.points
                poly = []
                for point in points:
                    poly.append(list(pyproj.transform(self.projin, self.projout, point[0], point[1])))
                self.writer.poly(parts=[poly])
        return self.writer

    # def write_shpfile(self, ):
    #     if self.reader.shape
    # #     self.reader
    # #     self.save(pathwrite)


def write_shpfile(shapeType=1, coords=None, poly_parts=None, path=None):

    shpfile = shapefile.Writer(shapeType=shapeType)

    if shapeType == 1:
        for coord in coords:
            if len(coord) == 2:
                shpfile.point(coord[0], coord[1])
            else:
                shpfile.point(coord[0], coord[1], coord[2])
    elif shapeType == 5:
        shpfile.field('ID', 'C', '40')
        for (id, part) in enumerate(poly_parts):
            shpfile.poly(parts=part)
            shpfile.record(str(id))

    if path is not None:
        shpfile.save(path)

    return shpfile

def shapes_in_polygons(shpfile_shapes, shpfile_polygons, zaxis=False):

    if shpfile_shapes.shapeType == 1 or shpfile_shapes.shapeType == 3:
        shapes = shapefile.Writer(shapeType=shpfile_shapes.shapeType)
        if len(shpfile_shapes.fields) > 0:
            for field in shpfile_shapes.fields:
                shapes.field(field[0], field[1], field[2], field[3])

        for i, shape in enumerate(shpfile_shapes.shapes()):
            point = slgeo.Point(shape.points[0])
            for poly in shpfile_polygons.shapes():
                if all([len(x) > 2 for x in poly.points]):
                    poly = [[x, y] for x, y, _, _ in poly.points]
                    if slgeo.Polygon(poly).contains(point):
                        if zaxis:
                            shapes.point(x=shape.points[0][0], y=shape.points[0][1], z=shape.points[0][2])
                        else:
                            shapes.point(x=shape.points[0][0], y=shape.points[0][1])
                        if isinstance(shpfile_shapes.records, list) is False:
                            shapes.records.append(shpfile_shapes.records()[i])
                elif slgeo.Polygon(poly.points[0]).contains(point):
                    if zaxis:
                        shapes.point(x=shape.points[0][0], y=shape.points[0][1], z=shape.points[0][2])
                    else:
                        shapes.point(x=shape.point[0][0], y=shape.point[0][1])
                    if isinstance(shpfile_shapes.records, list) is False:
                        shapes.records.append(shpfile_shapes.records()[i])

                for s in shapes.shapes():
                    s.shapeType = shpfile_shapes.shapeType

        return shapes
    else:
        pass


# if shpfile_shapes.shapeType == 1 or shpfile_shapes.shapeType == 3:
#     shapes = shapefile.Writer(shapeType=shpfile_shapes.shapeType)
#     if len(shpfile_shapes.fields) > 0:
#         for field in shpfile_shapes.fields:
#             shapes.field(field[0], field[1], field[2], field[3])
#
#     for i, shape in enumerate(shpfile_shapes.shapes()):
#         point = slgeo.Point(shape.points[0])
#         for poly in shpfile_polygons.shapes():
#             if all([len(x) > 2 for x in poly.points]):
#                 poly = [[x, y] for x, y, _, _ in poly.points]
#                 if slgeo.Polygon(poly).contains(point):
#                     shapes.point(x=shape.points[0][0], y=shape.points[0][1])
#                     if isinstance(shpfile_shapes.records, list) is False:
#                         shapes.record(shpfile_shapes.records()[i])
#             elif slgeo.Polygon(poly.points[0]).contains(point):
#                 shapes.point(x=shape.point[0][0], y=shape.point[0][1])
#
#     for s in shapes.shapes():
#         s.shapeType = shpfile_shapes.shapeType
#
#     if path is not None:
#         shapes.save(path)
#
#     return shapes
# else:
#     pass







        # def get_patches(self):
    #     if self.reader.shapeType == 5 or self.writer.shapeType == 5:
    #         self.read_shpfile()
    #         patches = []
    #         if self.projin is not None and self.projout is not None:
    #             self.transform_shapes()
    #             for shape in self.writer.shapes():
    #                 if all([len(x) > 2 for x in shape.points]):
    #                     poly = [[x, y] for x, y, _, _ in shape.points]
    #                     patches.append(mplpatch.Polygon(poly, closed=True, fill=False))
    #                 else:
    #                     patches.append(mplpatch.Polygon(shape.points))
    #         else:
    #             for shape in self.reader.shapes():
    #                 patches.append(mplpatch.Polygon(shape.points))
    #         return patches
    #     else:
    #         pass






# def get_patches(self):
#     if self.reader.shapeType == 5 or self.writer.shapeType == 5:
#         self.read_shpfile()
#         patches = []
#         if self.projin is not None and self.projout is not None:
#             self.transform_shapes()
#             for shape in self.writer.shapes():
#                 if all([len(x) > 2 for x in shape.points]):
#                     poly = [[x, y] for x, y, _, _ in shape.points]
#                     patches.append(mplpatch.Polygon(poly, closed=True, fill=False))
#                 else:
#                     patches.append(mplpatch.Polygon(shape.points))
#         else:
#             for shape in self.reader.shapes():
#                 patches.append(mplpatch.Polygon(shape.points))
#         return patches
#     else:
#         pass




# def coords_in_polygon(shpfile_points, shpfile_polygons, projection=None):
#
#     coords = []
#
#     try:
#         sf_points = shapefile.Reader(shpfile_points)
#         sf_polygons = shapefile.Reader(shpfile_polygons)
#     except shapefile.ShapefileException:
#         sf_points = shpfile_points
#         sf_polygons = shpfile_polygons
#
#     for shape in sf_points.shapes():
#         if projection is not None:
#             x, y = shape.points[0]
#             point = Point(pyproj.transform(projection, op.WGS84, x, y))
#         else:
#             point = Point(shape.points[0])
#         for poly in sf_polygons.shapes():
#             if all([len(x) > 2 for x in poly.points]):
#                 poly = [[x, y] for x, y, _, _ in poly.points]
#                 if Polygon(poly).contains(point):
#                     coords.append(point.coords[0])
#             elif Polygon(poly.points[0]).contains(point):
#                 coords.append(point.coords[0])
#     return coords