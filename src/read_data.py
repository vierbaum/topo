from bs4 import BeautifulSoup
from bs4.element import NavigableString
from tqdm import tqdm
import pickle
import threading
import itertools
import os
import pathlib
import block
import numpy as np
import projections.wagner
import geopandas as gpd
import shapely
from concurrent.futures import ThreadPoolExecutor
from time import sleep


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
            resolution=(80, 80),
        )
        current_block.load_image()
        current_block.export_as_pickle(f"{dataset_path}/80x80/{filename}")
        blocks.append(current_block)


def read_blocks_from_pickle(resolution_dir: pathlib.Path) -> list[block.Block]:
    blocks = []
    for path in tqdm(os.listdir(resolution_dir)):
        b = block.Block()
        b.read_from_pickle(resolution_dir / path)
        blocks.append(b)
    return blocks

def project_blocks(blocks, projection):
    for b in tqdm(blocks):
        projection(b)

def export_polygon(src_polygon, export_path, area_threashhold):
    dst_point_array =[]
    for long, lat in zip(src_polygon.exterior.xy[0], src_polygon.exterior.xy[1]):
        x, y = projections.wagner.angle_to_wagner(np.radians(long), np.radians(lat))
        if isinstance(x, float) and isinstance(y, float):
            dst_point_array.append([x,y])

    #square
    #if len(dst_point_array) <= 5:
    #    return

    dst_polygon = shapely.Polygon(dst_point_array)
    try:
        if dst_polygon.area < area_threashhold:
            return

        with open(export_path, "wb") as f:
            pickle.dump(dst_polygon, f)
    except Exception:
        print(dst_point_array)

def export_coastlines(coastlines_file="data/coastlines/land_polygons.shp", export_dir="data/coastlines/polygons"):
    gdf = gpd.read_file(coastlines_file)
    coastlines = gdf["geometry"]
    for i, src_polygon in tqdm(enumerate(coastlines)):
        export_polygon(src_polygon, f"{export_dir}/polygon{i}.pickle", 0)

def export_tile(blocks, tile, x, y, size):
    for c_block in tqdm(blocks):
        assert c_block.x_size != None
        assert c_block.y_size != None
        c_block.scale_z(1 / 30)
        projections.wagner.wagner_tile(
            topo_data=c_block.scaled_image.T,
            x_offset=c_block.x_off,
            y_offset=c_block.y_off,
            x_scale=c_block.x_size / c_block.scaled_image.shape[0],
            y_scale=c_block.y_size / c_block.scaled_image.shape[1],
            raster_x=block.Block.raster_x,
            raster_y=block.Block.raster_y,
            tile_arr=tile,
            tile_x=x,
            tile_y=y,
            total_x_res=tile.shape[0] * size[0],
            total_y_res=tile.shape[1] * size[1]
        )

if __name__ == "__main__":
    #export_coastlines()
    dataset_path = pathlib.Path("data/aw3d30")
    pixel_res = (4096, 4096)
    tile = np.zeros(pixel_res)

    block.Block.raster_x = 1296000
    block.Block.raster_y = 604800

    #read_xml(dataset_path=dataset_path, xml_file="AW3D30_global.vrt", num_threads=15)

    blocks = read_blocks_from_pickle(dataset_path / "360x360" / "AW3D30_global")

    export_tile(blocks, tile, 5, 5, (20, 10))

    print("EXPORTING")
    block.Block.world = tile
    block.Block.export_world_to_dat(64, 64)
