#!/usr/bin/python
from bs4 import BeautifulSoup
from bs4.element import NavigableString
from tqdm import tqdm
import threading
import itertools
import pathlib
from block import Block
import shutil
import os
from argparse import ArgumentParser


def read_xml(
    dataset_path: pathlib.Path,
    output_path:pathlib.Path, xml_filepath: pathlib.Path,
    num_threads: int,
    resolution: tuple[int, int]
) -> None:
    """Read in xml and distribute to threads."""
    # reading in xml file
    with open(xml_filepath, "r") as f:
        vrt_data = BeautifulSoup(f.read(), "xml")

    assert hasattr(vrt_data, "VRTDataset")
    dataset = vrt_data.VRTDataset

    assert dataset != None

    raster_x_size = dataset.get("rasterXSize")
    raster_y_size = dataset.get("rasterYSize")

    assert isinstance(raster_x_size, str)
    assert isinstance(raster_y_size, str)

    try:
        Block.raster_x = int(raster_x_size)
    except Exception:
        raise RuntimeError("failed to convert rasterXSize")
    try:
        Block.raster_y = int(raster_y_size)
    except Exception:
        raise RuntimeError("failed to convert rasterySize")

    # blocks are SimpleSource objects
    sources = vrt_data.find_all("SimpleSource")

    source_chunk = [itertools.islice(sources, i, len(sources), num_threads)
                    for i in range(num_threads)]
    threads = [
        threading.Thread(
            target=read_blocks_from_tif,
            args=(dataset_path, chunk, resolution, output_path)
        ) for i, chunk in enumerate(source_chunk)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def read_blocks_from_tif(
    dataset_path: pathlib.Path,
    sources: list[NavigableString],
    resolution: tuple[int, int],
    output_path: pathlib.Path
) -> None:
    """Generate blocks of data."""

    # transfering to Block class
    for source in tqdm(sources):

        assert source != None

        assert hasattr(source, "SourceFilename")

        filename = source.SourceFilename.contents[0]
        filepath = dataset_path / str(filename)

        assert hasattr(source, "SrcRect")
        x_size = int(source.SrcRect.get("xSize"))
        y_size = int(source.SrcRect.get("ySize"))

        assert hasattr(source, "DstRect")
        x_off = int(source.DstRect.get("xOff"))
        y_off = int(source.DstRect.get("yOff"))

        assert x_size % resolution[0] == 0
        assert y_size % resolution[1] == 0
        current_block = Block(
            filepath=filepath,
            x_off=x_off,
            y_off=y_off,
            x_size=x_size,
            y_size=y_size,
            resolution=resolution,
        )
        current_block.load_image()
        current_block.export_as_pickle(output_path / filename)

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="downsampledata",
        description="downsample tif data"
    )
    parser.add_argument("-i", "--vrt_path")
    parser.add_argument("--threads")
    parser.add_argument("--res")

    args = parser.parse_args()
    if args.vrt_path:
        vrt_path = pathlib.Path(args.vrt_path)
    else:
        vrt_path = pathlib.Path("data/aw3d30/AW3D30_global.vrt")
    assert vrt_path.is_file()

    assert args.res != None
    res = int(args.res)
    output_path = vrt_path.parent / f"{res}x{res}"
    if output_path.exists():
        shutil.rmtree(output_path)

    os.makedirs(output_path / "AW3D30_global")

    if args.threads:
        threads = int(args.threads)
    else:
        threads = 1

    read_xml(
        dataset_path=vrt_path.parent,
        output_path=output_path,
        xml_filepath=vrt_path,
        resolution=(res, res),
        num_threads=threads
    )
