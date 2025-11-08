import numpy as np
import math
from tifffile import imread
import pathlib
import pickle
from numba import njit
from typing import Callable
from tqdm import tqdm


class Block:
    """
    Store the data of a topographic block.
    """

    raster_x: int
    raster_y: int
    world: np.array

    def __init__(
        self, filepath: pathlib.Path = pathlib.Path(""),
        x_off: int | None = None,
        y_off: int | None = None,
        x_size: int | None = None,
        y_size: int | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> None:
        """Initialize self."""
        self.filepath = filepath

        if x_off is not None:
            self.x_off = x_off
        if y_off is not None and y_size is not None:
            self.y_off = self.raster_y - y_off - y_size

        self.x_size = x_size
        self.y_size = y_size

        if isinstance(resolution, list) and len(resolution) != 2:
            raise ValueError("Resolution needs to be of lenght 2")

        if resolution:
            self.scaled_image = np.zeros(resolution, dtype=np.int16)

    def export_projection(self, projection: Callable, exit: Callable) -> None:
        """
        Write the block to the world in north-up projection.
        """
        if exit(self):
            return

        x_scale = self.x_size / self.scaled_image.shape[0]
        y_scale = self.y_size / self.scaled_image.shape[1]

        half_x = self.world.shape[0] / 2
        half_y = self.world.shape[1] / 2

        for x in range(self.scaled_image.shape[0]):
            for y in range(self.scaled_image.shape[1]):
                x_, y_ = projection(self.x_off + x * x_scale, self.y_off +
                                    y * y_scale, self.raster_x, self.raster_y)

                x_ = round((x_ + 1) * half_x)
                y_ = round((y_ + 1) * half_y)

                x_ = min(x_, self.world.shape[0] - 1)
                y_ = min(y_, self.world.shape[1] - 1)

                self.world[x_, y_] = self.scaled_image[y, x]

    @classmethod
    def export_array_to_dat(cls, array: np.array, filepath: str) -> None:
        """
        Export a np array to a .dat file.

        array
            array to be exported

        filepath
            path to dat file
        """

        if np.max(array) == 0:
            return

        dat_rows = []
        for y in range(array.shape[1]):
            row = " ".join(str(array[x, y]) for x in range(array.shape[0]))
            dat_rows.append(row)

        with open(filepath, "w") as file:
            file.write("\n".join(dat_rows))

    @classmethod
    def export_world_to_dat(cls, chunks_x: int, chunks_y: int) -> None:
        """
        Export world to equally sized chunks.

        chunks_x
            number of chunks in x-direction

        chunks_y
            number of chunks in y-direction
        """
        reshaped_size = cls.world.shape // np.array((chunks_x, chunks_y))

        for x in tqdm(range(chunks_x)):
            for y in range(chunks_y):
                cls.export_array_to_dat(
                    cls.world[
                        x * reshaped_size[0]:(x + 1) * reshaped_size[0] + 1,
                        y * reshaped_size[1]:(y + 1) * reshaped_size[1] + 1,
                    ],
                    f"worlddata/world{x},{y}"
                )

    def read_from_pickle(self, path: pathlib.Path) -> None:
        """
        Read in Block form pickle file.

        path
            path of pickle file
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.filepath = data["filepath"]
        self.x_off = data["x_off"]
        self.y_off = data["y_off"]
        self.x_size = data["x_size"]
        self.y_size = data["y_size"]
        self.scaled_image = data["data"]

    def export_as_pickle(self, path: str) -> None:
        """
        Export block data to a pickle file.

        path
            filepath
        """
        data = {
            "filepath": self.filepath,
            "x_off": self.x_off,
            "y_off": self.y_off,
            "x_size": self.x_size,
            "y_size": self.y_size,
            "data": self.scaled_image
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def export_as_dat(self, filepath: str) -> None:
        """
        Export the block to an SCAD-readable format.

        Args
            filepath: str: file destination

        Raises
            RuntimeError if scaled_image not set.
        """

        if not hasattr(self, "scaled_image"):
            raise RuntimeError("scaled_image not initialized")

        dat_rows = []
        for x in range(self.scaled_image.shape[0]):
            row = " ".join(str(self.scaled_image[x, y]) for y in range(self.scaled_image.shape[1]))
            dat_rows.append(row)

        with open(filepath, "w") as file:
            file.write("\n".join(dat_rows))

    def scale_z(self, scale: float) -> None:
        "Scale the image by a factor in z axis"
        self.scaled_image *= scale

    def offset_z(self, offset):
        "Offset the image by a factor in z axis"
        self.scaled_image += offset

    def load_image(self) -> None:
        """
        Load an image from a tif file.
        The image is scaled by average to the given resolution.

        Raises
            RuntimeError if x_size, y_size is not set.
        """

        image = imread(self.filepath)

        if self.x_size is None:
            raise RuntimeError("x_size not initialized")
        if self.y_size is None:
            raise RuntimeError("y_size not initialized")

        # scaling factors for np scaling
        x_scale = self.x_size // self.scaled_image.shape[0]
        y_scale = self.y_size // self.scaled_image.shape[1]

        # autopep8: off
        self.scaled_image = image.reshape(-1, y_scale, image.shape[0])\
            .mean(axis=1)\
            .reshape(self.scaled_image.shape[0], self.scaled_image.shape[1], x_scale).mean(axis=2)
        # autopep8: on
        self.scaled_image = np.rot90(self.scaled_image, k=3)


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
        180W = 0pi
        0W/E = pi
        180E = 2pi

    longitude:
        90S = 0pi
        0N/S = pi/2
        90N = pi
    """

    latitude = (x * 2) / raster_x * math.pi
    longitude = y / raster_y * math.pi

    return latitude, longitude


@njit
def projection_north_up(x: float, y: float, raster_x: int, raster_y: int) -> tuple[float, float]:
    """
        Map position of point in equidistant projection to north-up projection.

        x
            x position
        y
            y position

        raster_x
            total size of raster in x-direction
        raster_y
            total size of raster in y-direction
    """
    latitude, longitude = unmercator(x, y, raster_x, raster_y)
    alpha = longitude - math.pi / 2
    beta = latitude
    scale = np.cos(alpha)
    x = np.sin(beta)
    y = np.cos(beta)

    return x * scale, y * scale
