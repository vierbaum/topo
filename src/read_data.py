from tqdm import tqdm
import pickle
import os
import pathlib
import block
import numpy as np
import projections.wagner
import geopandas as gpd
import shapely






def read_blocks_from_pickle(resolution_dir: pathlib.Path) -> list[block.Block]:
    blocks = []
    for path in tqdm(os.listdir(resolution_dir)):
        b = block.Block()
        b.read_from_pickle(resolution_dir / path)
        blocks.append(b)
    return blocks

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
            topo_data=c_block.scaled_image,
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
    pixel_res = (1024, 1024)
    tile = np.zeros(pixel_res)

    block.Block.raster_x = 1296000
    block.Block.raster_y = 604800

    #read_xml(dataset_path=dataset_path, xml_file="AW3D30_global.vrt", num_threads=15)

    blocks = read_blocks_from_pickle(dataset_path / "120x120" / "AW3D30_global")

    export_tile(blocks, tile, 5, 5, (20, 10))

    print("EXPORTING")
    block.Block.world = tile
    block.Block.export_world_to_dat(32, 32)
