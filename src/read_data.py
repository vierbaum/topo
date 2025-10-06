from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np
from tifffile import imread

TOPO_PATH = "data/aw3d30/"
VRT_PATH = TOPO_PATH + "AW3D30_global.vrt"

class Block:
    """
    Store the data of a topographic block.
    """

    def __init__(self, filepath: str, x_off: int, y_off: int, x_size: int, y_size: int, z_scale: int, resolution: tuple[int, int], z_offset: int) -> None:
        """Initialize self."""
        self.filepath = filepath
        self.x_off = x_off
        self.y_off = y_off
        self.x_size = x_size
        self.y_size = y_size
        self.z_scale = z_scale
        #self.scaled_image = np.zeros((resolution[0], resolution[1]), dtype=np.int16)
        self.scale_x = x_size // resolution[0]
        self.scale_y = y_size // resolution[1]
        self.resolution = resolution
        self.z_offset = z_offset

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

def read_blocks() -> list[Block]:
    """Generate blocks of data."""
    # reading in xml file
    with open(VRT_PATH, "r") as f:
        vrt_file = f.read()
    vrt_data = BeautifulSoup(vrt_file, "xml")

    # blocks are SimpleSource objects
    sources = vrt_data.find_all("SimpleSource")

    blocks = []

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
            z_scale=1,
            resolution=(x_size // 100, y_size // 100),
            z_offset=0
        )
        current_block.read_image()
        blocks.append(current_block)

    return blocks

blocks = read_blocks()
