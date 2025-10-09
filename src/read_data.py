from bs4 import BeautifulSoup
from bs4.element import NavigableString
from tqdm import tqdm
import numpy as np
from tifffile import imread
import threading
import itertools
import math

NUM_THREADS = 15
TOPO_PATH = "data/aw3d30/"
VRT_PATH = TOPO_PATH + "AW3D30_global.vrt"

class Block:
    """
    Store the data of a topographic block.
    """

    raster_x: int
    raster_y: int


    def __init__(self, filepath: str, x_off: int, y_off: int, x_size: int, y_size: int, z_scale: int, resolution: tuple[int, int], z_offset: int) -> None:
        """Initialize self."""
        self.filepath = filepath

        self.x_off = x_off
        # data is south up
        self.y_off = 180 - y_off

        self.x_size = x_size
        self.y_size = y_size

        self.scale_x = x_size // resolution[0]
        self.scale_y = y_size // resolution[1]
        self.z_scale = z_scale
        self.z_offset = z_offset

        self.resolution = resolution

    def unmeractor(self, x: int, y: int) -> tuple[float, float]:
        """Return angular coordinates on globe for any given x, y coordinates."""
        latitude = 2 * (x + self.x_off) / self.raster_x * math.pi + math.pi
        longitude = (y + self.y_off) / self.raster_y * math.pi

        return latitude, longitude

    def export_as_dat(self, filepath: str) -> None:
        data = ""
        for x in range(self.resolution[0]):
            for y in range(self.resolution[1]):
                data += "%s "%self.scaled_image[x, y]
            data += "\n"

        with open(filepath, "w") as file:
            file.write(data)

    def read_image(self) -> None:
        image = imread(self.filepath)

        # full image
        if self.scale_x == self.scale_y == 1:
            self.scaled_image = image
            return

        reshaped = image.reshape(self.resolution[0], self.scale_x, self.resolution[1], self.scale_y)
        self.scaled_image = reshaped.mean(axis=(1,3))
        # flipping it along x-axis
        self.scaled_image = np.flip(self.scaled_image, axis=0)
        self.scaled_image *= 1 / self.z_scale
        self.export_as_dat(f"heightdata/data{self.x_off // self.x_size},{self.y_off // self.y_size}.dat")

def read_xml() -> list[Block]:
    """Read in xml and distribute to threads"""
    # reading in xml file
    with open(VRT_PATH, "r") as f:
        vrt_data = BeautifulSoup(f.read(), "xml")
    Block.raster_x = int(vrt_data.VRTDataset.get("rasterXSize"))
    Block.raster_y = int(vrt_data.VRTDataset.get("rasterYSize"))

    # blocks are SimpleSource objects
    sources = vrt_data.find_all("SimpleSource")
    source_chunk = [itertools.islice(sources, i, len(sources), NUM_THREADS) for i in range(NUM_THREADS)]
    returned_blocks = [[] for _ in range(NUM_THREADS)]
    threads = [threading.Thread(target = read_blocks, args = (chunk, returned_blocks[i])) for i, chunk in enumerate(source_chunk)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return list(itertools.chain.from_iterable(returned_blocks))


def read_blocks(sources: list[NavigableString], blocks: list):
    """Generate blocks of data."""

    # transfering to Block class
    for source in tqdm(sources):
        filename = source.SourceFilename.contents[0]

        x_size = int(source.SrcRect.get("xSize"))
        y_size = int(source.SrcRect.get("ySize"))

        x_off = int(source.DstRect.get("xOff"))
        y_off = int(source.DstRect.get("yOff"))

        # 1:1 scale
        current_block = Block(
            filepath=TOPO_PATH + str(filename),
            x_off=x_off,
            y_off=y_off,
            x_size=x_size,
            y_size=y_size,
            z_scale=60,
            resolution=(x_size // 60, y_size // 60),
            z_offset=0
        )
        current_block.read_image()
        blocks.append(current_block)

if __name__ == "__main__":
    blocks = read_xml()
