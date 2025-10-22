import numpy as np
import math
from tifffile import imread
import pathlib
import pickle


class Block:
    """
    Store the data of a topographic block.
    """

    raster_x: int
    raster_y: int

    def __init__(
        self, filepath: pathlib.Path = pathlib.Path(""),
        x_off: int | None = None,
        y_off: int | None = None,
        x_size: int | None = None,
        y_size: int | None = None,
        z_scale: int | None = None,
        resolution: tuple[int, int] | None = None,
        z_offset: int | None = None
    ) -> None:
        """Initialize self."""
        self.filepath = filepath

        self.x_off = x_off
        self.y_off = y_off

        self.x_size = x_size
        self.y_size = y_size

        if isinstance(resolution, list) and len(resolution) != 2:
            raise ValueError("Resolution needs to be of lenght 2")

        self.z_scale = z_scale
        self.z_offset = z_offset

        if resolution:
            self.scaled_image = np.zeros(resolution, dtype=np.int16)

    def unmeractor(self, x: int, y: int) -> tuple[float, float]:
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

        latitude = (x + self.x_off) / self.raster_x * 2 * math.pi
        longitude = (y + self.y_off) / self.raster_y * math.pi

        return latitude, longitude

    def read_from_pickle(self, path: pathlib.Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.filepath = data["filepath"]
        self.x_off = data["x_off"]
        self.y_off = data["y_off"]
        self.x_size = data["x_size"]
        self.y_size = data["y_size"]
        self.z_scale = data["z_scale"]
        self.z_offset = data["z_offset"]
        self.scaled_image = data["data"]

    def export_as_pickle(self, path: str) -> None:
        data = {
            "filepath": self.filepath,
            "x_off": self.x_off,
            "y_off": self.y_off,
            "x_size": self.x_size,
            "y_size": self.y_size,
            "z_scale": self.z_scale,
            "z_offset": self.z_offset,
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

    def load_image(self) -> None:
        """
        Load an image from a tif file.
        The image is scaled by average to the given resolution.

        Raises
            RuntimeError if x_size, y_size, z_scale is not set.
        """

        image = imread(self.filepath)

        if self.x_size is None:
            raise RuntimeError("x_size not initialized")
        if self.y_size is None:
            raise RuntimeError("y_size not initialized")
        if self.z_scale is None:
            raise RuntimeError("z_scale not initialized")

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

        # apply z-scaling
        self.scaled_image *= 1 / self.z_scale
