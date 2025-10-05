from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np

TOPO_PATH = "../data/aw3d30/"
VRT_PATH = TOPO_PATH + "AW3D30_global.vrt"

class Block:
    """
    Store the data of a topographic block.
    """

    def __init__(self, filename: str, x_off: int, y_off: int, resolution: tuple[int, int]) -> None:
        """Initialize self."""
        self.filename = filename
        self.x_off = x_off
        self.y_off = y_off
        self.scaled_image = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint16)

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
        blocks.append(Block(str(filename), x_off, y_off, (x_size, y_size)))

    return blocks
