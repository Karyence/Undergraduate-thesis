import xarray as xr
import numpy as np
import os
import glob
import time
import warnings
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 配置参数 
# =============================================================================
ERA5_DIR = "/data/Climate/ERA5/ERA5-LAND/"
REF_GRID_FILE = os.path.expanduser("~/bisheshuju/GEBCO/GEBCO_2025_YRD_1km.nc")
OUTPUT_FILE = os.path.expanduser("~/bisheshuju/ERA5_LAND/ERA5_wind_speed_1km_downscaled.nc")

FILE_PATTERNS = ["era_land_20250[1-9]*.nc", "era_land_20251[0-2]*.nc"]

# =============================================================================
# 统计对比工具函数
# =============================================================================
def get_resolution(coord_array):
    if len(coord_array) > 1:
        return round(abs(float(coord_array[1] - coord_array[0])), 4)
    return 0.0

def calc_detailed_stats(da):
    """计算风速数据的关键统计指标"""
    lon_coord = 'lon' if 'lon' in da.coords else 'longitude'
    lat_coord = 'lat' if 'lat' in da.coords else 'latitude'

    lon_vals = da[lon_coord].values
    lat_vals = da[lat_coord].values

    return {
        "经度分辨率(°)": f"{get_resolution(lon_vals):.4f}",
        "纬度分辨率(°)": f"{get_resolution(lat_vals):.4f}",
        "经度网格点数": f"{len(lon_vals)}",
        "纬度网格点数": f"{len(lat_vals)}",
        "时间序列长度(小时)": f"{da.shape[0]}",
        "小时风速最小值(m/s)": f"{float(da.min().compute()):.4f}",
        "小时风速最大值(m/s)": f"{float(da.max().compute()):.4f}",
        "小时区域平均风速(m/s)": f"{float(da.mean().compute()):.4f}"
    }

def print_final_comparison(stats_before, stats_after):
    """打印降尺度前后数据统计对比报告"""
    print("\n" + "="*90)
    print("📊 ERA5 Wind Speed (10m风速) 降尺度前后详细数据对比报告".center(86))
    print("="*90)
    header = f"{'统计指标':<22} | {'降尺度后数据 (1km)':<25} | {'原始 ERA5 数据 (0.1°)':<25}"
    print(header)
    print("-" * 90)
    for key in stats_before.keys():
        print(f"{key:<22} | {str(stats_after[key]):<29} | {str(stats_before[key]):<25}")
    print("="*90 + "\n")

# =============================================================================
# 2. 核心处理逻辑
# =============================================================================
def _rename_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_dict = {}
    if "valid_time" in ds.coords: rename_dict["valid_time"] = "time"
    if "longitude" in ds.coords: rename_dict["longitude"] = "lon"
    if "latitude" in ds.coords: rename_dict["latitude"] = "lat"
    return ds.rename(rename_dict) if rename_dict else ds

def process_wind_downscaling():
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='48GB')
    client = Client(cluster)
    print(f"🚀 已启动满血计算集群！监控地址: {client.dashboard_link}")
    
    print("🚀 开始执行 ERA5 10m风速矢量合成与小时级降尺度")
    start_time = time.time()

    # 1. 加载地形基准网格
    ds_grid = xr.open_dataset(REF_GRID_FILE)
    target_lon = ds_grid['lon'].values
    target_lat = ds_grid['lat'].values
    buf = 0.3 
    min_lon, max_lon = target_lon.min() - buf, target_lon.max() + buf
    min_lat, max_lat = target_lat.min() - buf, target_lat.max() + buf

    # 2. 扫描并加载数据
    file_list = sorted([f for p in FILE_PATTERNS for f in glob.glob(os.path.join(ERA5_DIR, p)) if os.path.isfile(f)])
    datasets = []
    # 保留 u10 和 v10
    drop_vars = ['t2m', 'skt', 'sp', 'd2m', 'tp'] 
    
    for idx, fp in enumerate(file_list):
        try:
            with xr.open_dataset(fp, engine='netcdf4', drop_variables=drop_vars) as ds:
                ds = _rename_coords(ds)
                lat_slice = slice(max_lat, min_lat) if ds.lat[0] > ds.lat[-1] else slice(min_lat, max_lat)
                ds = ds.sel(lon=slice(min_lon, max_lon), lat=lat_slice)
                datasets.append(ds)
        except Exception:
            continue

    # 引入时间块，按 24小时(一天) 进行分块，加速后续的按天压缩写入
    ds_concat = xr.concat(datasets, dim="time", join="override")
    ds_concat = ds_concat.chunk({"time": 24, "lat": -1, "lon": -1})
    
    # 3. 核心算法：矢量合成全风速
    print("[处理中] 正在执行 u10 和 v10 的风速矢量合成 (延迟计算)...")
    ws_hourly = np.sqrt(ds_concat['u10']**2 + ds_concat['v10']**2)
    
    # 提取 2025 全年数据
    ws_hourly = ws_hourly.sel(time=slice("2025-01-01", "2025-12-31"))
    stats_before = calc_detailed_stats(ws_hourly)

    # 4. 执行双线性插值
    print("[处理中] 映射至 1km 网格并执行空间插值 (延迟计算)...")
    ws_interp = ws_hourly.interp(
        lon=target_lon, lat=target_lat, 
        method='linear',
        kwargs={"fill_value": "extrapolate"} 
    )
    
    # 物理下限约束：风速绝对不可能为负数
    ws_interp = ws_interp.clip(min=0.0)
    
    # 对比分析
    stats_after = calc_detailed_stats(ws_interp)
    print_final_comparison(stats_before, stats_after)

    # 5. 保存结果
    print("[存储中] 正在导出 NetCDF 文件...")
    ws_interp.name = 'wind_speed'
    ws_interp.attrs = {
        'units': 'm s**-1',
        'long_name': '10 metre wind speed',
        'description': 'Hourly wind speed calculated from u10 and v10 vectors, bilinearly downscaled to 1km grid.'
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    encoding = {
        'wind_speed': {
            'zlib': True, 
            'complevel': 1, 
            'dtype': 'float32', 
            'chunksizes': (24, len(target_lat), len(target_lon)),
            '_FillValue': -9999.0
        }
    }
    
    with ProgressBar():
        ws_interp.to_netcdf(OUTPUT_FILE, encoding=encoding, engine='netcdf4', compute=True)
        
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"✅ 处理完成！耗时: {time.time() - start_time:.1f} 秒。文件大小: {size_mb:.2f} MB")

if __name__ == "__main__":
    process_wind_downscaling()