import numpy as np
import math
from tifffile import imread
import pathlib
import pickle
from numba import njit


class Block:
    """
    Store the data of a topographic block.
    """

    raster_x: int
    raster_y: int
    export: np.array

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

        if x_off != None:
            self.x_off = x_off
        if y_off != None and y_size != None:
            self.y_off = self.raster_y - y_off - y_size

        self.x_size = x_size
        self.y_size = y_size

        if isinstance(resolution, list) and len(resolution) != 2:
            raise ValueError("Resolution needs to be of lenght 2")

        if resolution:
            self.scaled_image = np.zeros(resolution, dtype=np.int16)

    def unmercator(self, x: int, y: int) -> tuple[float, float]:
        """
        Return angular coordinates on globe for any given x, y coordinates.
        latitude:
            180W = 0pi
            0W/E = pi
            180E = 2pi

        longitude:
            90S = 0pi
            0N/S = pi/2
            90N = pi
        """

        if self.x_off is None:
            raise RuntimeError("x_off needs to be of type int, not None")
        if self.y_off is None:
            raise RuntimeError("y_off needs to be of type int, not None")

        latitude = ((x + self.x_off) * 2) / self.raster_x * math.pi
        longitude = (y + self.y_off) / self.raster_y * math.pi

        return latitude, longitude

    def export_projection(self):
        if self.y_off < self.raster_y / 2 or self.x_off < 2 / 4 * self.raster_x or self.x_off > 3 / 4 * self.raster_x:
            return

        x_scale = self.x_size / self.scaled_image.shape[0]
        y_scale = self.y_size / self.scaled_image.shape[1]

        half_x = self.world.shape[0] / 2
        half_y = self.world.shape[1] / 2

        for x in range(self.scaled_image.shape[0]):
            for y in range(self.scaled_image.shape[1]):
                x_, y_ = self.projection_north_up(x * x_scale, y * y_scale)

                x_ = (x_ + 1) * half_x
                y_ = (y_ + 1) * half_y

                x_ = min(int(x_), self.world.shape[0] - 1)
                y_ = min(int(y_), self.world.shape[1] - 1)

                self.world[x_, y_] = self.scaled_image[y,x]

    def projection_north_up(self, x, y):
        latitude, longitude = self.unmercator(x, y)
        alpha = longitude - math.pi / 2
        beta = latitude
        scale = np.cos(alpha)
        x = np.sin(beta)
        y = np.cos(beta)

        return x * scale, y * scale


    def read_from_pickle(self, path: pathlib.Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.filepath = data["filepath"]
        self.x_off = data["x_off"]
        self.y_off = data["y_off"]
        self.x_size = data["x_size"]
        self.y_size = data["y_size"]
        self.scaled_image = data["data"]

    def export_as_pickle(self, path: str) -> None:
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

    def scale_z(self, scale):
        self.scaled_image *= scale

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

        reshaped = image.reshape(
            self.scaled_image.shape[0],
            x_scale,
            self.scaled_image.shape[1],
            y_scale)
        self.scaled_image = reshaped.mean(axis=(1, 3))

        # flipping it along x-axis
        self.scaled_image = np.flip(self.scaled_image, axis=0)
