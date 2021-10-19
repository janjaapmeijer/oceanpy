# https://pypi.python.org/pypi/utm

import pyproj
import numpy as np

__all__ = ['haversine', 'rotatexy', 'decompose_cartesian_to_natural']

# geodetic defining parameters (wgs84: http://earth-info.nga.mil/GandG/publications/tr8350.2/wgs84fin.pdf)
#              model               major (m)     flattening
ellipsoids = {'WGS-84':           (6378137.0,    1 / 298.257223563),
              'GRS-80':           (6378137.0,    1 / 298.257222101),
              'Airy (1830)':      (6377563.396,  1 / 299.3249646),
              'Intl 1924':        (6378388.0,    1 / 297.0),
              'Clarke (1880)':    (6378249.145,  1 / 293.465),
              'GRS-67':           (6378160.0,    1 / 298.25),
              }
for model in ellipsoids.keys():
    a = ellipsoids[model][0]
    f = ellipsoids[model][1]
    b = (1 - f) * a
    e = (f * (2 - f))**(1 / 2)
    ellipsoids[model] = (a, b, f, e)


### CONVERSIONS ###
WGS84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
UTM33 = pyproj.Proj("+proj=utm +zone=33U, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
ETRS89 = pyproj.Proj('+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs')
# SWEREF99 13 30
SWE99 = pyproj.Proj("+proj=tmerc +lat_0=0 +lon_0=13.5 +k=1 +x_0=150000 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
# SWEREF99 TM
SWE99TM = pyproj.Proj('+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
# OSM = pyproj.Proj('+ellps=WGS84 +proj=tmerc +lat_0=0 +lon_0=015d48.377m +k=1.0000056 +x_0=1500064.1 +y_0=-668.0')

UTM54 = pyproj.Proj('EPSG:32754')
UTM55 = pyproj.Proj('EPSG:32755')

def swe2utm(xswe, yswe, zswe=None):

    if zswe == None:
        xutm, yutm = pyproj.transform(SWE99, UTM33, xswe, yswe)
        return xutm, yutm
    else:
        xutm, yutm, zutm = pyproj.transform(SWE99, UTM33, xswe, yswe, zswe)
        return xutm, yutm, zutm

# def dms_to_dd(deg, min, sec=None):
#     ''' Converting from Degrees, Minutes and Seconds to Decimal Degrees '''
#     if sec is None:
#         if deg < 0 or deg.endswidth('S'):
#             dd = - (min/60) + deg
#         else:
#             dd = (min/60) + deg
#     else:
#         if deg < 0:
#             dd = - (sec/3600) - (min/60) + deg
#         else:
#             dd = (sec/3600) + (min/60) + deg
#
#     return dd


### ROTATIONS ###
def rotatexyz(x0, y0, z0, lstx, lsty, lstz, theta, dn=None):

    import numpy as np

    # translation matrix
    trans = np.matrix([[1,0,0,x0],
                       [0,1,0,y0],
                       [0,0,1,z0],
                       [0,0,0,1]])

    # inverse of translation matrix
    inv_trans = np.matrix([[1,0,0,-x0],
                           [0,1,0,-y0],
                           [0,0,1,-z0],
                           [0,0,0,1]])

    # rotation matrix
    rot = np.matrix([[np.cos(theta), -np.sin(theta), 0, 0],
                     [np.sin(theta), np.cos(theta), 0, 0],
                     [0,0,1,0],
                     [0,0,0,1]])

    # combined (inverse)translation and rotation matrix (does not work)
    #  rot_mat = np.matrix([[np.cos(alpha[1]), -np.sin(alpha[1]), 0, (p*np.cos(alpha[1]) - q*np.sin(alpha[1]) - p)],
    #                      [np.sin(alpha[1]), np.cos(alpha[1]), 0, (p*np.sin(alpha[1]) + q*np.cos(alpha[1]) - q)],
    #                      [0,0,1,0],
    #                      [0,0,0,1]])

    if dn is not None:
        xyzvector = [np.matrix([[lstx[i]],[lsty[i]],[lstz[i]],[1]]) for i in range(0,len(lstx),dn)]
    else:
        xyzvector = [np.matrix([[lstx[i]],[lsty[i]],[lstz[i]],[1]]) for i in range(0,len(lstx))]

    xprime = []
    yprime = []
    zprime = []

    for a in xyzvector:
        xyzprimeinv = np.dot(inv_trans,a)
        xyzprimerot = np.dot(rot, xyzprimeinv)
        xyzprimetrans = np.dot(trans, xyzprimerot)
        xprime.append(float(xyzprimetrans[0]))
        yprime.append(float(xyzprimetrans[1]))
        zprime.append(float(xyzprimetrans[2]))

    return xprime, yprime, zprime

def rotatexyz_pnt(x0, y0, z0, x, y, z, theta):

    import numpy as np

    # translation matrix
    trans = np.matrix([[1,0,0,x0],
                       [0,1,0,y0],
                       [0,0,1,z0],
                       [0,0,0,1]])

    # inverse of translation matrix
    inv_trans = np.matrix([[1,0,0,-x0],
                           [0,1,0,-y0],
                           [0,0,1,-z0],
                           [0,0,0,1]])

    # rotation matrix
    rot = np.matrix([[np.cos(theta), -np.sin(theta), 0, 0],
                     [np.sin(theta), np.cos(theta), 0, 0],
                     [0,0,1,0],
                     [0,0,0,1]])

    # combined (inverse)translation and rotation matrix (does not work)
    #  rot_mat = np.matrix([[np.cos(alpha[1]), -np.sin(alpha[1]), 0, (p*np.cos(alpha[1]) - q*np.sin(alpha[1]) - p)],
    #                      [np.sin(alpha[1]), np.cos(alpha[1]), 0, (p*np.sin(alpha[1]) + q*np.cos(alpha[1]) - q)],
    #                      [0,0,1,0],
    #                      [0,0,0,1]])

    xyzvector = np.matrix([[x],[y],[z],[1]])

    xyzprimeinv = np.dot(inv_trans,xyzvector)
    xyzprimerot = np.dot(rot, xyzprimeinv)
    xyzprimetrans = np.dot(trans, xyzprimerot)
    xprime = float(xyzprimetrans[0])
    yprime = float(xyzprimetrans[1])
    zprime = float(xyzprimetrans[2])

    return (xprime, yprime, zprime)

def rotatexy(x0, y0, lstx, lsty, theta, dn=None):

    import numpy as np

    # translation matrix
    trans = np.matrix([[1,0,x0],
                       [0,1,y0],
                       [0,0,1]])

    # inverse of translation matrix
    inv_trans = np.matrix([[1,0,-x0],
                           [0,1,-y0],
                           [0,0,1]])

    # rotation matrix
    rot = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0,0,1]])

    if dn is not None:
        xyvector = [np.matrix([[lstx[i]],[lsty[i]],[1]]) for i in range(0,len(lstx),dn)]
    else:
        xyvector = [np.matrix([[lstx[i]],[lsty[i]],[1]]) for i in range(0,len(lstx))]

    # xprime = []
    # yprime = []
    coords = []

    for a in xyvector:
        xyprimeinv = np.dot(inv_trans,a)
        xyprimerot = np.dot(rot, xyprimeinv)
        xyprimetrans = np.dot(trans, xyprimerot)
        # xprime.append(float(xyprimetrans[0]))
        # yprime.append(float(xyprimetrans[1]))
        coords.append((float(xyprimetrans[0]),float(xyprimetrans[1])))
    # return xprime, yprime
    return coords

def rotatexy_pnt(x0, y0, x, y, theta):

    import numpy as np

    # translation matrix
    trans = np.matrix([[1,0,x0],
                       [0,1,y0],
                       [0,0,1]])

    # inverse of translation matrix
    inv_trans = np.matrix([[1,0,-x0],
                           [0,1,-y0],
                           [0,0,1]])

    # rotation matrix
    rot = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0,0,1]])

    xyvector = [np.matrix([[x],[y],[1]])]

    for a in xyvector:
        xyprimeinv = np.dot(inv_trans,a)
        xyprimerot = np.dot(rot, xyprimeinv)
        xyprimetrans = np.dot(trans, xyprimerot)
        xprime = float(xyprimetrans[0])
        yprime = float(xyprimetrans[1])

    return (xprime, yprime)


### LOCATIONS ###
def geodetic2ecef(lon, lat, h, radian=False):
    '''
    Earth Centered, Earth Fixed (ECEF)
    http://www.colorado.edu/geography/gcraft/notes/datum/gif/llhxyz.gif
    1.22 http://what-when-how.com/the-3-d-global-spatial-data-model/the-gsdm-the-global-spatial-data-model-gsdm-defined-part-2/
    :param lon:
    :param lat:
    :param h:
    :param radian:
    :return:
    '''

    global a, e

    lamb = np.asarray(lon) if radian else np.radians(np.asarray(lon))
    phi = np.asarray(lat) if radian else np.radians(np.asarray(lat))
    h = np.asarray(h)

    # check if dimensions theta and phi are equal
    if lamb.size != phi.size != h.size:
        raise ValueError("input parameters must have the same dimensions")

    # determine radius of the earth
    aa = np.ones(phi.shape) * a
    N = aa / (1 - e**2 * np.sin(phi)**2)**(1 / 2)

    return ((N + h) * np.cos(lamb) * np.cos(phi),
            (N + h) * np.sin(lamb) * np.cos(phi),
            (N * (1 - e**2) + h) * np.sin(phi))

def ecef2enu(x, y, z, lon0=0., lat0=0., h0=0., radian=False):
    '''
    http://www.navipedia.net/index.php/Transformations_between_ECEF_and_ENU_coordinates

    :param x:
    :param y:
    :param z:
    :param lon0:
    :param lat0:
    :param h0:
    :param radian:
    :return:
    '''

    global a, e

    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    lamb, phi = (lon0, lat0) if radian else (np.radians(lon0), np.radians(lat0))

    # check if dimensions theta and phi are equal
    if x.size != y.size != z.size:
        raise ValueError("input parameters must have the same dimensions")
    if not all(list(map(lambda x: np.isscalar(x), [lamb, phi, h0]))):
        raise ValueError("lon0, lat0 and h0 must be scalars")

    x0, y0, z0 = geodetic2ecef(lon0, lat0, h0)

    R = np.array([[-np.sin(lamb), np.cos(lamb), 0],
                  [-np.cos(lamb) * np.sin(phi), -np.sin(lamb) * np.sin(phi), np.cos(phi)],
                  [np.cos(lamb) * np.cos(phi), np.sin(lamb) * np.cos(phi), np.sin(phi)]])

    enu = np.dot(R, np.array([x - x0, y - y0, z - z0]))

    return tuple(enu)

def llh2enu(lon, lat, h, lon0=0., lat0=0., h0=0., radian=False):
    return ecef2enu(*geodetic2ecef(lon, lat, h, radian=radian), lon0=lon0, lat0=lat0, h0=h0)

# http://codegists.com/code/ecef-to-enu-python/
# https://gis.stackexchange.com/questions/82998/trasformation-from-ecef-to-enu
# ftp://ftp.soest.hawaii.edu/nosal/ORE603/Lectures_FullPage/ORE603_04EquationsOfMotion.pdf
# http://www.navipedia.net/index.php/Transformations_between_ECEF_and_ENU_coordinates
# http://what-when-how.com/the-3-d-global-spatial-data-model/the-gsdm-the-global-spatial-data-model-gsdm-defined-part-2/
# https://en.wikipedia.org/wiki/Geographic_coordinate_conversion


### DISTANCES ###
def haversine(lon, lat, h=0.):
    '''
    Great circle distance between longitude and latitude coordinates

    Parameters
    ----------
    :param lon: longitude coordinates
    :param lat: latitude coordinates

    Returns
    -------
    :return: distance and bearing
    '''

    lamb = np.radians(np.asarray(lon))
    phi = np.radians(np.asarray(lat))

    dlamb = np.diff(lamb)
    dphi = np.diff(phi)

    i = (slice(0, -1))
    ip1 = (slice(1, None))

    if np.all(h == 0.):
        h=0.
    else:
        h = (h[ip1] - h[i]) / 2

    a = np.sin(dphi / 2)**2 + np.cos(phi[i]) * np.cos(phi[ip1]) * np.sin(dlamb / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371e3

    distance = c * (R + h)

    bearing = np.arctan2(np.sin(dlamb) * np.cos(phi[ip1]),
                         np.cos(phi[i]) * np.sin(phi[ip1]) - np.sin(phi[i]) * np.cos(phi[ip1]) * np.cos(dlamb))
    bearing = (np.rad2deg(bearing) + 360) % 360

    return distance, bearing

def vincenty(lon, lat, ellps='WGS-84', iterlim=20):
    '''
    Geodesic distance between longitude and latitude coordinates

    Parameters
    ----------
    :param lon:
    :param lat:
    :param radian:
    :param iterlim:

    Returns
    -------
    :return:
    '''

    global ellipsoids
    a, _, f, _ = ellipsoids[ellps]

    lamb = np.radians(np.asarray(lon))
    phi = np.radians(np.asarray(lat))

    distance = []
    bearing = []

    for i in range(lamb.size - 1):

        L = lamb[i+1] - lamb[i]
        # reduced latitude
        U1 = np.arctan((1 - f) * np.tan(phi[i]))
        U2 = np.arctan((1 - f) * np.tan(phi[i+1]))

        sinU1, cosU1 = np.sin(U1), np.cos(U1)
        sinU2, cosU2 = np.sin(U2), np.cos(U2)

        Lambda, LambdaP = L, 2 * np.pi
        iteration = 0
        while abs(Lambda - LambdaP) > 1e-12 and iterlim > iteration >= 0:
            iteration += 1
            sinLambda, cosLambda = np.sin(Lambda), np.cos(Lambda)
            sinSigma = np.sqrt((cosU2 * sinLambda) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)
            if (sinSigma == 0):
                raise ValueError('co-incident points')
            cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
            Sigma = np.arctan2(sinSigma, cosSigma)
            sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
            cosSqAlpha = 1 - sinAlpha ** 2
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
            if np.isnan(cos2SigmaM):
                cos2SigmaM = 0  # equatorial line: cosSqAlpha = 0
            C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
            LambdaP = Lambda
            Lambda = L + (1 - C) * f * sinAlpha * \
                         (Sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM ** 2)))
        if iteration == iterlim:
            raise ValueError('formula failed to converge')
        uSq = cosSqAlpha * ((a ** 2 - b ** 2) / b ** 2)
        A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
        B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
        deltaSigma = B * sinSigma * (cos2SigmaM + (B / 4) * (cosSigma * (-1 + 2 * cos2SigmaM ** 2) -
                                                             (B / 6) * cos2SigmaM * (-3 + 4 * sinSigma ** 2) * (
                                                             -3 + 4 * cos2SigmaM ** 2)))
        Alpha = np.arctan2(cosU2 * sinLambda, cosU1 * sinU2 - sinU1 * cosU2 * cosLambda)
        # Alpha2 = np.arctan2(cosU1 * sinLambda, -sinU1 * cosU2 + cosU1 * sinU2 * cosLambda)

        distance.append(b * A * (Sigma - deltaSigma))
        bearing.append((np.degrees(Alpha) + 360) % 360)
        # bearing1, bearing2 = r2d(Alpha1), r2d(Alpha2)
        # bearing1, bearing2 = (bearing1 + 360) % 360, (bearing2 + 360) % 360

    return np.asarray(distance), np.asarray(bearing)

def bearing_to_standard(bearing):
    return np.deg2rad((90 - bearing) % 360)

def decompose_cartesian_to_natural(u, v, angle, bearing=False):

    if (type(u) == list) or (type(v) == list) or (type(angle) == list):
        u, v, angle = np.array(u), np.array(v), np.array(angle)

    if bearing:
        theta = bearing_to_standard(angle)
    else:
        theta = angle

    print('theta : ', theta)
    ori = theta.copy()

    # ut = u*0-1
    # vn = v*0-1

    # q1 = (theta > 0) & (theta <= np.pi/2)
    # ut[q1] = u[q1] * np.cos(ori[q1]) + v[q1] * np.sin(ori[q1])
    # vn[q1] = - u[q1] * np.sin(ori[q1]) + (v[q1] * np.cos(ori[q1]))

    # q4 = (theta > 3*np.pi/2) & (theta <= 2*np.pi)
    # ori[q4] = theta[q4] - (3*np.pi/2)
    # ut[q4] = u[q4] * np.cos(ori[q4]) + v[q4] * np.sin(ori[q4])
    # vn[q4] = u[q4] * np.sin(ori[q4]) + v[q4] * np.cos(ori[q4])
    # print('ut-decompose : ', ut)


    ut = u * np.cos(ori) + v * np.sin(ori)
    vn = u * np.sin(ori) + v * np.cos(ori)




    # xpos = (theta > np.pi/2) & (theta <= 3*np.pi/2)
    # ypos = (theta > np.pi) & (theta <= 2*np.pi)
    # print(ori)
    # ori[xpos] = theta[xpos] + np.pi
    # ori[ypos] = theta[ypos]
    # ori[q4] = np.pi/2 + theta[q4]
    # print(bearing_to_standard(angle)-theta)



    # print(u * np.cos(ori), v * np.sin(ori))
    # print(u * np.sin(ori), v * np.cos(ori))

    return ut, vn, ori
