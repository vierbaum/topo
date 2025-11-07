from numba import njit
import numpy as np


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

    latitude = (x / raster_x * 2 * np.pi) - np.pi  # % (2 * np.pi)
    longitude = y / raster_y * np.pi - np.pi / 2

    return latitude, longitude


@njit
def angle_to_wagner(latitude: float, longitude: float) -> tuple[float, float]:
    # lambda: latitude
    # phi: longitude

    m_1 = 0.9063
    m_2 = 1
    n = 1 / 3
    k = 1.4660
    c_x = 5.3344
    c_y = 2.4820
    # wagner(-pi, 0)
    x_correction = 2 * 2.6671999999999993 + 0.001
    # wagner(-pi, -pi/2)
    y_correction = 2 * 1.4452060370320987 + 0.001

    psi = np.arcsin(m_1 * np.sin(m_2 * longitude))
    delta = np.arccos(np.cos(n * latitude) * np.cos(psi))
    alpha = np.arccos(np.sin(psi) / np.sin(delta))
    x = c_x * np.sin(delta / 2) * np.sin(alpha) / x_correction
    y = c_y * np.sin(delta / 2) * np.cos(alpha) / y_correction

    if latitude < 0:
        return -x, y
    return x, y


@njit
def wagner_tile(
        topo_data,
        x_offset,
        y_offset,
        x_scale,
        y_scale,
        raster_x,
        raster_y,
        tile_arr,
        tile_x,
        tile_y,
        total_x_res,
        total_y_res,
):
    assert x_scale > 0
    assert y_scale > 0

    for x in range(topo_data.shape[0]):
        for y in range(topo_data.shape[1]):
            latitude, longitude = unmercator(
                x * x_scale + x_offset, y * y_scale + y_offset, raster_x, raster_y)
            x_, y_ = angle_to_wagner(latitude, longitude)
            #print(x, y, latitude, longitude, x_, y_)

            # x,y in [-0.5, 0.5], scaling to total res
            x_ *= total_x_res - 1
            x_ += total_x_res / 2
            y_ *= total_y_res - 1
            y_ += total_y_res / 2
            # offseting to 0
            x_ = round(x_)
            x_ -= tile_x * tile_arr.shape[0]
            y_ = round(y_)
            y_ -= tile_y * tile_arr.shape[1]

            is_in_tile_x = 0 <= x_ < tile_arr.shape[0]
            is_in_tile_y = 0 <= y_ < tile_arr.shape[1]
            if is_in_tile_x and is_in_tile_y:
                tile_arr[x_, y_] = topo_data[x, y]



@njit
def wagner(
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
            latitude, longitude = unmercator(
                x * x_scale + x_offset, y * y_scale + y_offset, raster_x, raster_y)
            x_, y_ = angle_to_wagner(latitude, longitude)
            try:
                x_ *= projection.shape[0] - 1
                x_ += projection.shape[0] / 2
                y_ *= projection.shape[1] - 1
                y_ += projection.shape[1] / 2
                x_ = round(x_)
                y_ = round(y_)

                x_ = round((x * x_scale + x_offset) / raster_x * projection.shape[0])
                y_ = round((y * y_scale + y_offset) / raster_y * projection.shape[1])
                if 0 <= x_ < projection.shape[0] and 0 <= y_ < projection.shape[1]:
                    projection[x_, y_] = topo_data[x, y]
            except Exception:
                print(x, y, latitude, longitude, x_, y_)
