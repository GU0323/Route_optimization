from math import acos, atan2, pi, cos, sin

def jinbuk(lat1,lon1,lat2,lon2):

    lat1 = lat1

    lon1 = lon1

    lat2 = lat2

    lon2 = lon2

    y1 = lat1 * pi / 180

    y2 = lat2 * pi / 180

    x1 = lon1 * pi / 180

    x2 = lon2 * pi / 180

    Y = sin(x2 - x1) * cos(y2)
    X = cos(y1) * sin(y2) - sin(y1) * cos(y2) * cos(x2 - x1)
    theta = atan2(Y, X)

    result = (theta * 180 / pi + 360) % 360

    return result


