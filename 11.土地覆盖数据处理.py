import xarray as xr
import numpy as np
import os
import zipfile
import shutil
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import gc
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')

# =============================================================================
# 1. 配置参数
# =============================================================================
CONF = {
    "LC_ZIP_DIR": "/data/Environment/1米国土资源/7709370东",
    "TARGET_GRID": os.path.expanduser("~/bisheshuju/GEBCO/GEBCO_2025_YRD_1km.nc"),
    "OUTPUT_NC": os.path.expanduser("~/bisheshuju/LandCover/YRD_LandCover_Fractions_1km_Final.nc"),
    
    "TARGET_ZIPS": [
        "East_Shanghai.zip",
        "East_Anhui.zip",
        "East_Jiangsu.zip",
        "East_Zhejiang.zip",
        "East_Shandong.zip",
        "East_Fujian.zip",
        "East_Jiangxi.zip"
    ],
    
    "LC_CLASSES": {
        1: "traffic", 
        2: "forest",  
        3: "shrubland", 
        4: "grassland", 
        5: "cropland",  
        6: "building",  
        7: "barren",    
        9: "water",     
        10: "wetland"   
    },
    
    "DOWNSAMPLE_RES_M": 10,
    "ORIGIN_RES_M": 1,
    "USE_RAM_DISK": True,
    "RAM_DISK_PATH": "/dev/shm",
    "REPROJECT_THREADS": 8,
    "BLOCK_SIZE": 8192,
    "MAX_WORKERS": 4,
    "CHUNK_SIZE": 256,
    "TEMP_ROOT": "/tmp/sinolc_yrd"
}

# =============================================================================
# 核心处理函数
# =============================================================================
def process_single_tif(tif_path, target_transform, target_crs, target_shape, downsample_ratio):
    if not os.path.exists(tif_path): return None
    try:
        with rasterio.open(tif_path) as src:
            tif_left, tif_bottom, tif_right, tif_top = src.bounds
            tar_left, tar_bottom, tar_right, tar_top = rasterio.transform.array_bounds(*target_shape, target_transform)
            
            if (tif_right < tar_left or tif_left > tar_right or tif_bottom > tar_top or tif_top < tar_bottom):
                return None
            
            src_width, src_height = src.width, src.height
            dst_width = int(np.ceil(src_width * downsample_ratio))
            dst_height = int(np.ceil(src_height * downsample_ratio))
            downsampled_data = np.zeros((dst_height, dst_width), dtype=np.uint8)
            downsampled_transform = src.transform * rasterio.Affine.scale(1/downsample_ratio, 1/downsample_ratio)
            
            n_blocks_x = max(1, int(np.ceil(src_width / CONF["BLOCK_SIZE"])))
            n_blocks_y = max(1, int(np.ceil(src_height / CONF["BLOCK_SIZE"])))
            
            for y_block in range(n_blocks_y):
                for x_block in range(n_blocks_x):
                    x_off = x_block * CONF["BLOCK_SIZE"]
                    y_off = y_block * CONF["BLOCK_SIZE"]
                    win_width = min(CONF["BLOCK_SIZE"], src_width - x_off)
                    win_height = min(CONF["BLOCK_SIZE"], src_height - y_off)
                    window = Window(x_off, y_off, win_width, win_height)
                    
                    block_data = src.read(1, window=window, out_shape=(int(win_height * downsample_ratio), int(win_width * downsample_ratio)), resampling=Resampling.nearest)
                    dst_y_off, dst_x_off = int(y_off * downsample_ratio), int(x_off * downsample_ratio)
                    dst_h, dst_w = block_data.shape
                    downsampled_data[dst_y_off:dst_y_off+dst_h, dst_x_off:dst_x_off+dst_w] = block_data
        
        class_ids = list(CONF["LC_CLASSES"].keys())
        tif_fractions = np.zeros((len(class_ids), target_shape[0], target_shape[1]), dtype=np.float32)
        
        for i, class_id in enumerate(class_ids):
            binary_mask = (downsampled_data == class_id).astype(np.float32)
            if np.sum(binary_mask) == 0: continue
            temp_dest = np.zeros(target_shape, dtype=np.float32)
            reproject(
                source=binary_mask, destination=temp_dest, 
                src_transform=downsampled_transform, src_crs=src.crs, 
                dst_transform=target_transform, dst_crs=target_crs, 
                resampling=Resampling.average, num_threads=CONF["REPROJECT_THREADS"]
            )
            tif_fractions[i] = temp_dest
        
        del downsampled_data
        if 'binary_mask' in locals(): del binary_mask
        gc.collect()
        return tif_fractions
    except Exception:
        return None

def extract_zip_to_permanent_temp(zip_path, temp_root):
    province_name = os.path.basename(zip_path).replace(".zip", "").replace("East_", "")
    province_temp = os.path.join(temp_root, province_name)
    os.makedirs(province_temp, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if not [f for f in zip_ref.namelist() if f.lower().endswith(('.tif', '.tiff'))]: return []
        zip_ref.extractall(province_temp)
    return [os.path.join(root, file) for root, _, files in os.walk(province_temp) for file in files if file.lower().endswith(('.tif', '.tiff'))]

def main():
    print("="*80)
    print("🚀 长三角1m土地覆盖 → 1km占比 ")
    print("="*80)
    
    temp_root = CONF["TEMP_ROOT"]
    if CONF["USE_RAM_DISK"] and os.path.exists(CONF["RAM_DISK_PATH"]):
        temp_root = os.path.join(CONF["RAM_DISK_PATH"], "sinolc_yrd")
    os.makedirs(temp_root, exist_ok=True)
    
    print("\n[1/4] 加载目标1km网格 (GEBCO)...")
    ds_target = xr.open_dataset(CONF["TARGET_GRID"], chunks={"lat": CONF["CHUNK_SIZE"], "lon": CONF["CHUNK_SIZE"]})
    target_lon, target_lat = ds_target["lon"].values, ds_target["lat"].values
    target_height, target_width = len(target_lat), len(target_lon)
    target_shape = (target_height, target_width)
    
    target_bounds = (np.min(target_lon), np.min(target_lat), np.max(target_lon), np.max(target_lat))
    target_transform = rasterio.transform.from_bounds(*target_bounds, target_width, target_height)
    
    class_ids = list(CONF["LC_CLASSES"].keys())
    class_names = [CONF["LC_CLASSES"][k] for k in class_ids]
    total_fractions = np.zeros((len(class_ids), target_height, target_width), dtype=np.float32)
    
    print("\n[2/4] 解压所有ZIP到持久化临时目录...")
    all_tif_paths = []
    for zip_name in CONF["TARGET_ZIPS"]:
        zip_path = os.path.join(CONF["LC_ZIP_DIR"], zip_name)
        if os.path.exists(zip_path): 
            all_tif_paths.extend(extract_zip_to_permanent_temp(zip_path, temp_root))
    
    print(f"\n[3/4] 多进程处理 {len(all_tif_paths)} 个TIF文件...")
    with ProcessPoolExecutor(max_workers=CONF["MAX_WORKERS"]) as executor:
        futures = {executor.submit(process_single_tif, path, target_transform, "EPSG:4326", target_shape, CONF["ORIGIN_RES_M"] / CONF["DOWNSAMPLE_RES_M"]): path for path in all_tif_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
            tif_fractions = future.result()
            if tif_fractions is not None: total_fractions += tif_fractions
    
    print("\n[4/4] 数据后处理与维度对齐...")
    total_per_grid = np.sum(total_fractions, axis=0)
    valid_mask = total_per_grid > 0.001
    
    if np.sum(valid_mask) == 0:
        final_fractions = np.zeros_like(total_fractions)
    else:
        total_per_grid[~valid_mask] = 1.0
        final_fractions = total_fractions / total_per_grid[np.newaxis, :, :]
        final_fractions = np.clip(final_fractions, 0, 1)
        final_fractions[:, ~valid_mask] = np.nan
        
    # 纬度方向自动对齐
    if target_lat[0] < target_lat[-1]:
        print("   -> 侦测到纬度数组为升序，自动执行南北翻转以匹配真实地理坐标！")
        final_fractions = final_fractions[:, ::-1, :]
    
    print("\n[5/5] 保存结果文件...")
    ds_output = xr.Dataset()
    for i, class_name in enumerate(class_names):
        da = xr.DataArray(
            final_fractions[i, :, :], dims=["lat", "lon"], coords={"lat": target_lat, "lon": target_lon},
            attrs={"units": "fraction (0-1)", "long_name": f"Area fraction of {class_name.capitalize()} in 1km Grid"}
        )
        ds_output[f"{class_name}_frac"] = da.chunk({"lat": CONF["CHUNK_SIZE"], "lon": CONF["CHUNK_SIZE"]})
    
    os.makedirs(os.path.dirname(CONF["OUTPUT_NC"]), exist_ok=True)
    encoding = {var: {"zlib": True, "complevel": 6, "dtype": "float32", "_FillValue": np.nan} for var in ds_output.data_vars}
    ds_output.to_netcdf(CONF["OUTPUT_NC"], encoding=encoding)
    
    shutil.rmtree(temp_root)
    print(f"✅ 结果已保存至: {CONF['OUTPUT_NC']}")

if __name__ == "__main__":
    main()