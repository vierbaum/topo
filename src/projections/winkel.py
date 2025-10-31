from numba import njit
import numpy as np
import math


@njit
def unmercator(x: float, y: float, raster_x: int, raster_y: int) -> tuple[float, float]:
    """
    Return angular coordinates on globe for any given x, y coordinates.
    x
        x position
    y
        y position

    raster_x
        total size of raster in x-direction
    raster_y
        total size of raster in y-direction

    latitude:
        180W = -pi
        0W/E = 0
        180E = pi

    longitude:
        90S = -pi/2
        0N/S = 0
        90N = pi/2
    """

    latitude = (x / raster_x * 2 * np.pi) - np.pi# % (2 * np.pi)
    longitude = y / raster_y * np.pi - np.pi / 2

    return latitude, longitude


@njit
def angle_to_winkel(latitude: float, longitude: float) -> tuple[float, float]:
    alpha = np.arccos(np.cos(longitude) * np.cos(latitude / 2))
    phi_1 = np.arccos(2/np.pi)

    alpha_si = np.sin(alpha) / alpha if alpha != 0 else 1

    #x = latitude / (2 * np.pi)# + 2 * np.cos(longitude) * np.sin(latitude / 2) / alpha_si) / 2
    #y = longitude / np.pi# + np.sin(longitude) / alpha_si) / 2
    x = (latitude + 2 * np.cos(longitude) * np.sin(latitude / 2) / alpha_si) / (4 * np.pi)
    y = (longitude + np.sin(longitude) / alpha_si) / (2 * np.pi)
    return x, y

@njit
def winkel(
        topo_data: np.array,
        projection: np.array,
        x_offset: int,
        y_offset: int,
        x_scale: float,
        y_scale: float,
        raster_x: int,
        raster_y: int):
    for x in range(topo_data.shape[0]):
        for y in range(topo_data.shape[1]):
            latitude, longitude = unmercator(x * x_scale + x_offset, y * y_scale + y_offset, raster_x, raster_y)

            x_, y_ = angle_to_winkel(latitude, longitude)
            #print(f"lat\t{math.degrees(latitude)}\nlong\t{math.degrees(longitude)}\nx_\t{x_}\ny_\t{y_}")
            x_ *= projection.shape[0] - 1
            x_ += projection.shape[0] / 2
            y_ *= projection.shape[1] - 1
            y_ += projection.shape[1] / 2
            x_ = round(x_)
            y_ = round(y_)

            projection[x_, y_] = topo_data[y, x]
