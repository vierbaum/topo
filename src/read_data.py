from bs4 import BeautifulSoup
from bs4.element import NavigableString
from tqdm import tqdm
import threading
import itertools
import os
import pathlib
import block
import numpy as np
import pickle


def read_xml(dataset_path: pathlib.Path, xml_file: str, num_threads: int) -> list[block.Block]:
    """Read in xml and distribute to threads."""
    xml_filepath = dataset_path / xml_file
    if not xml_filepath.exists():
        raise FileNotFoundError(f"{xml_filepath} is not a valid path.")

    # reading in xml file
    with open(xml_filepath, "r") as f:
        vrt_data = BeautifulSoup(f.read(), "xml")

    if not hasattr(vrt_data, "VRTDataset"):
        raise RuntimeError("failed to read VRTDataset")
    dataset = vrt_data.VRTDataset

    if not dataset:
        raise RuntimeError("failed to read dataset")

    raster_x_size = dataset.get("rasterXSize")
    raster_y_size = dataset.get("rasterYSize")

    if not isinstance(raster_x_size, str):
        raise RuntimeError("failed to read rasterXSize")
    if not isinstance(raster_y_size, str):
        raise RuntimeError("failed to read rasterYSize")

    try:
        block.Block.raster_x = int(raster_x_size)
    except Exception:
        raise RuntimeError("failed to convert rasterXSize")
    try:
        block.Block.raster_y = int(raster_y_size)
    except Exception:
        raise RuntimeError("failed to convert rasterySize")

    # blocks are SimpleSource objects
    sources = vrt_data.find_all("SimpleSource")

    source_chunk = [itertools.islice(sources, i, len(sources), num_threads)
                    for i in range(num_threads)]
    returned_blocks = [[] for _ in range(num_threads)]
    threads = [
        threading.Thread(
            target=read_blocks_from_tif,
            args=(dataset_path, chunk, returned_blocks[i])
        ) for i, chunk in enumerate(source_chunk)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return list(itertools.chain.from_iterable(returned_blocks))


def read_blocks_from_tif(dataset_path: pathlib.Path, sources: list[NavigableString], blocks: list):
    """Generate blocks of data."""

    # transfering to Block class
    for source in tqdm(sources):

        if not source:
            raise RuntimeError("source may not be empty.")

        if not hasattr(source, "SourceFilename"):
            raise RuntimeError("failed to read block filename.")

        filename = source.SourceFilename.contents[0]
        filepath = dataset_path / str(filename)

        if not hasattr(source, "SrcRect"):
            raise RuntimeError("failed to read block SrcRect.")
        x_size = int(source.SrcRect.get("xSize"))
        y_size = int(source.SrcRect.get("ySize"))

        if not hasattr(source, "DstRect"):
            raise RuntimeError("failed to read block DstRect.")
        x_off = int(source.DstRect.get("xOff"))
        y_off = int(source.DstRect.get("yOff"))

        current_block = block.Block(
            filepath=filepath,
            x_off=x_off,
            y_off=y_off,
            x_size=x_size,
            y_size=y_size,
            resolution=(x_size // 60, y_size // 60),
        )
        current_block.load_image()
        current_block.export_as_pickle(f"{dataset_path}/60x60/{filename}")
        blocks.append(current_block)


def read_blocks_from_pickle(resolution_dir: pathlib.Path) -> list[block.Block]:
    blocks = []
    for path in tqdm(os.listdir(resolution_dir)):
        b = block.Block()
        b.read_from_pickle(resolution_dir / path)
        blocks.append(b)
    return blocks


if __name__ == "__main__":
    dataset_path = pathlib.Path("data/aw3d30")
    # read_xml(dataset_path=dataset_path, xml_file="AW3D30_global.vrt", num_threads=15)
    block.Block.world = np.full((8192, 8192), 200)

    block.Block.raster_x = 1296000
    block.Block.raster_y = 604800
    # with open("world.pickle", "rb") as f:
    #    block.Block.world = pickle.load(f)
    blocks = read_blocks_from_pickle(dataset_path / "60x60" / "AW3D30_global")

    for c_block in tqdm(blocks):
        c_block.scale_z(1 / 60)
        c_block.offset_z(200)
        # c_block.export_projection()
        c_block.export_as_dat(
            f"heightdata/data{c_block.x_off // c_block.x_size},{c_block.y_off // c_block.y_size}")

    # with open("world.pickle", "wb") as f:
    #    pickle.dump(block.Block.world, f)

    # block.Block.export_world_to_dat(256, 256)
