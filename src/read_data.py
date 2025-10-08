from bs4 import BeautifulSoup
from bs4.element import NavigableString
from tqdm import tqdm
import numpy as np
from tifffile import imread
import threading
import itertools

NUM_THREADS = 15
TOPO_PATH = "data/aw3d30/"
VRT_PATH = TOPO_PATH + "AW3D30_global.vrt"

class Block:
    """
    Store the data of a topographic block.
    """

    def __init__(self, filepath: str, x_off: int, y_off: int, x_size: int, y_size: int, z_scale: int, resolution: tuple[int, int], z_offset: int) -> None:
        """Initialize self."""
        self.filepath = filepath

        self.x_off = x_off // x_size
        # data is south up
        self.y_off = 180 - (y_off // y_size)

        self.x_size = x_size
        self.y_size = y_size

        self.scale_x = x_size // resolution[0]
        self.scale_y = y_size // resolution[1]
        self.z_scale = z_scale
        self.z_offset = z_offset

        self.resolution = resolution

    def export_as_dat(self, filepath: str) -> None:
        data = ""
        for x in range(self.resolution[0]):
            for y in range(self.resolution[1]):
                # data is south up
                data += "%s "%self.scaled_image[self.resolution[0] - x - 1, y]
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
        self.scaled_image *= 1 / self.z_scale
        self.export_as_dat(f"heightdata/data{self.x_off},{self.y_off}.dat")

def read_xml() -> list[Block]:
    """Read in xml and distribute to threads"""
    # reading in xml file
    with open(VRT_PATH, "r") as f:
        vrt_file = f.read()
    vrt_data = BeautifulSoup(vrt_file, "xml")

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
            z_scale=100,
            resolution=(x_size // 100, y_size // 100),
            z_offset=0
        )
        current_block.read_image()
        blocks.append(current_block)

blocks = read_xml()
