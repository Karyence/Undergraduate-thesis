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
OUTPUT_FILE = os.path.expanduser("~/bisheshuju/ERA5_LAND/ERA5_tp_1km_downscaled.nc")

FILE_PATTERNS = ["era_land_20250[1-9]*.nc", "era_land_20251[0-2]*.nc"]

# =============================================================================
# 统计对比工具函数
# =============================================================================
def get_resolution(coord_array):
    if len(coord_array) > 1:
        return round(abs(float(coord_array[1] - coord_array[0])), 4)
    return 0.0

def calc_detailed_stats(da):
    """计算降水数据的关键统计指标"""
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
        "小时降水最小值(mm)": f"{float(da.min().compute()):.4f}", 
        "小时降水最大值(mm)": f"{float(da.max().compute()):.2f}",
        "小时均区域降水(mm)": f"{float(da.mean().compute()):.4f}"
    }

def print_final_comparison(stats_before, stats_after):
    """打印降尺度前后数据对比报告"""
    print("\n" + "="*90)
    print("📊 ERA5 tp (日总降水量) 降尺度前后详细数据对比报告".center(86))
    print("="*90)
    header = f"{'对比指标':<22} | {'降尺度后数据 (1km)':<25} | {'原始 ERA5 数据 (0.1°)':<25}"
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

def process_tp():
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='48GB')
    client = Client(cluster)
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
    drop_vars = ['t2m', 'skt', 'sp', 'd2m', 'u10', 'v10'] 
    
    for idx, fp in enumerate(file_list):
        try:
            with xr.open_dataset(fp, engine='netcdf4', drop_variables=drop_vars) as ds:
                ds = _rename_coords(ds)
                lat_slice = slice(max_lat, min_lat) if ds.lat[0] > ds.lat[-1] else slice(min_lat, max_lat)
                ds = ds.sel(lon=slice(min_lon, max_lon), lat=lat_slice)
                datasets.append(ds)
        except Exception:
            continue

    ds_concat = xr.concat(datasets, dim="time", join="override")
    ds_concat = ds_concat.chunk({"time": 24, "lat": -1, "lon": -1})
    
    # 3. 核心算法：差分还原独立小时降水量
    print("[处理中] 执行小时差分还原以解绑累积降水量...")
    tp_mm = ds_concat['tp'] * 1000.0
    tp_shifted = tp_mm.shift(time=1)
    tp_diff = tp_mm - tp_shifted
    
    # 每日 01:00 UTC 的累积值即为该小时降水量，其余时间采用差分值
    tp_hourly = xr.where(tp_mm['time'].dt.hour == 1, tp_mm, tp_diff)
    tp_hourly = tp_hourly.where(tp_hourly >= 0, 0.0)
    
    # 4. 聚合为日总降水量
    tp_hourly = tp_hourly.sel(time=slice("2025-01-01", "2025-12-31"))

    stats_before = calc_detailed_stats(tp_hourly)

    # 5. 执行双线性插值降尺度
    print("[处理中] 映射至 1km 网格并执行空间插值...")
    tp_interp = tp_hourly.interp(
        lon=target_lon, lat=target_lat, 
        method='linear',
        kwargs={"fill_value": "extrapolate"} 
    )
    
    # 插值后极值裁剪 (物理阈值约束)
    tp_interp = tp_interp.clip(min=0.0)
    
    # 对比分析
    stats_after = calc_detailed_stats(tp_interp)
    print_final_comparison(stats_before, stats_after)

    # 6. 保存结果
    print("[存储中] 正在导出 NetCDF 文件...")
    tp_interp.name = 'tp'
    tp_interp.attrs = {
        'units': 'mm',
        'long_name': 'Total hourly precipitation',
        'description': 'Hourly precipitation calculated by differencing cumulative values, resampled to daily sums, downscaled to 1km grid.'
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    encoding = {
        'tp': {
            'zlib': True, 
            'complevel': 1, 
            'dtype': 'float32', 
            'chunksizes': (24, len(target_lat), len(target_lon)),
            '_FillValue': -9999.0
        }
    }
    
    tp_interp.to_netcdf(OUTPUT_FILE, encoding=encoding, engine='netcdf4', compute=True)
        
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"✅ 处理完成！耗时: {time.time() - start_time:.1f} 秒。文件大小: {size_mb:.2f} MB")

if __name__ == "__main__":
    process_tp()