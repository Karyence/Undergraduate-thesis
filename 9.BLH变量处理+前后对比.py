import xarray as xr
import numpy as np
import os
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. 配置参数
# =============================================================================
INPUT_BLH_FILE = "/home/wangzonghan/ERA5-BLH/BLH.nc"
REF_GRID_FILE = os.path.expanduser("~/bisheshuju/GEBCO/GEBCO_2025_YRD_1km.nc")
OUTPUT_FILE = os.path.expanduser("~/bisheshuju/ERA5_LAND/ERA5_BLH_1km_downscaled.nc")
VAR_NAME = 'blh'

# =============================================================================
# 统计分析工具函数
# =============================================================================
def calc_data_stats(da, coord_lon, coord_lat):
    """计算时空维度信息及核心物理指标"""
    dims = da.shape
    lon_res = abs(float(da[coord_lon][1] - da[coord_lon][0]))
    lat_res = abs(float(da[coord_lat][1] - da[coord_lat][0]))
    
    data_min = float(da.min().compute())
    data_max = float(da.max().compute())
    data_mean = float(da.mean().compute())

    return {
        "维度": f"{dims[0]} × {dims[1]} × {dims[2]}",
        "经度分辨率(°)": round(lon_res, 6),
        "纬度分辨率(°)": round(lat_res, 6),
        "最小值(m)": round(data_min, 2),
        "最大值(m)": round(data_max, 2),
        "平均值(m)": round(data_mean, 2)
    }

def print_stats_compare(before_stats, after_stats):
    """打印降尺度前后数据统计对比报告"""
    print("\n" + "="*80)
    print("📊 BLH数据 插值前后统计对比表".center(80))
    print("="*80)
    header = f"{'统计指标':<20} | {'插值前(原始网格)':<28} | {'插值后(1km网格)':<28}"
    print(header)
    print("-"*80)
    for key in before_stats.keys():
        print(f"{key:<20} | {str(before_stats[key]):<28} | {str(after_stats[key]):<28}")
    print("="*80 + "\n")

# =============================================================================
# 2. 核心处理逻辑
# =============================================================================
def process_blh():
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='150GB')
    client = Client(cluster)
    print(f"🚀 已启动计算集群！监控地址: {client.dashboard_link}")
    print("🚀 开始执行 ERA5 边界层高度 (BLH) 小时级 1km 降尺度")

    # 1. 准备目标网格
    ds_grid = xr.open_dataset(REF_GRID_FILE)
    target_lon = ds_grid['lon'].values
    target_lat = ds_grid['lat'].values
    print(f"1. 目标网格: 0.01°分辨率(1km)，尺寸 {len(target_lat)} x {len(target_lon)}")

    # 2. 加载 BLH 原始数据并进行分块与初剪
    try:
        ds = xr.open_dataset(INPUT_BLH_FILE, engine='netcdf4', chunks={'valid_time': 24, 'time': 24})
        if 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})
        
        # 空间初步裁剪
        buf = 0.3
        lat_slice = slice(target_lat.max()+buf, target_lat.min()-buf) if ds.latitude[0] > ds.latitude[-1] else slice(target_lat.min()-buf, target_lat.max()+buf)
        ds = ds.sel(longitude=slice(target_lon.min()-buf, target_lon.max()+buf), latitude=lat_slice)
        
        da = ds[VAR_NAME]
        print(f"2. 原始BLH数据加载与剪裁成功: {da.shape} (Time, Lat, Lon)")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 3. 提取 2025 全年小时级数据 
    print("[处理中] 提取 2025 全年小时级数据...")
    da_hourly = da.sel(time=slice("2025-01-01", "2025-12-31"))
    da_hourly.attrs = {'units': 'm', 'long_name': 'Boundary Layer Height (Hourly)'}
    print(f"3. 小时级提取完成: {da_hourly.shape}")

    # 4. 执行双线性插值
    src_lon_name = 'longitude' if 'longitude' in da_hourly.coords else 'lon'
    src_lat_name = 'latitude' if 'latitude' in da_hourly.coords else 'lat'

    print("[处理中] 映射至 1km 网格并执行空间插值 (延迟计算)...")
    da_interp = da_hourly.interp(
        {src_lon_name: target_lon, src_lat_name: target_lat},
        method='linear',
        kwargs={"fill_value": "extrapolate"}
    )
    
    # 物理极值范围约束：BLH (边界层高度) 不可能为负数
    da_interp = da_interp.clip(min=0.0)

    # 坐标名称规范化
    if src_lon_name != 'lon' or src_lat_name != 'lat':
        da_interp = da_interp.rename({src_lon_name: 'lon', src_lat_name: 'lat'})
    print("4. 空间插值与物理范围约束处理完成")

    # 5. 生成统计对比分析
    print("5. 正在执行插值前后指标核算...")
    stats_before = calc_data_stats(da_hourly, src_lon_name, src_lat_name)
    stats_after = calc_data_stats(da_interp, 'lon', 'lat')
    print_stats_compare(stats_before, stats_after)

    # 6. 保存 NetCDF 结果
    da_interp = da_interp.astype(np.float32)
    
    da_interp.name = 'blh'
    da_interp.attrs['description'] = 'Hourly resampled to 1km using bilinear interpolation.'

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    encoding = {
        'blh': {
            'zlib': True, 
            'complevel': 1, 
            '_FillValue': -9999.0
        }
    }
    
    try:
        # 🔥 核心修改 4：暴力内存直写，断开并发写入锁
        print("[计算中] 正在将全量插值结果加载到内存 ...")
        with ProgressBar():
            da_final = da_interp.compute()
        
        print("[存储中] 内存加载完毕！正在单线程极速安全写入磁盘...")
        da_final.to_netcdf(OUTPUT_FILE, encoding=encoding, engine='netcdf4')
        
        size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"✅ 处理完成！文件大小: {size_mb:.2f} MB，最终维度: {da_final.shape}")
    except Exception as e:
        print(f"❌ 存储失败: {e}")

if __name__ == "__main__":
    process_blh()