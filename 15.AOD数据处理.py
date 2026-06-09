import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from datetime import datetime
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 核心配置参数
# =============================================================================
CONF = {
    "DATA_DIR": "/data/Satellite/AOD_YRD",  
    "TARGET_GRID": os.path.expanduser("~/bisheshuju/GEBCO/GEBCO_2025_YRD_1km.nc"),
    "OUTPUT_NC": os.path.expanduser("~/bisheshuju/AOD/YRD_AOD_Daily_1km_2025.nc"),
    
    # MAIAC VJ119A2 SDS 关键字
    "AOD_SDS": "Optical_Depth_055",  
    "QA_SDS": "AOD_QA",              
    "SCALE_FACTOR": 0.001,           
}

# =============================================================================
# 2. 质量控制 (QA) 解析函数
# =============================================================================
def apply_maiac_qa_mask(aod_arr, qa_arr, nodata_val):
    """基于 MAIAC QA 掩码执行位运算筛选 (晴空、无邻近云、非填充值)"""
    cloud_mask = qa_arr & 0b111                 # 提取 0-2 位 (云掩膜)
    adj_mask = (qa_arr >> 5) & 0b111            # 提取 5-7 位 (邻近云掩膜)
    
    valid_mask = (cloud_mask == 1) & (adj_mask == 0) & (aod_arr != nodata_val)
    
    aod_clean = np.where(valid_mask, aod_arr * CONF["SCALE_FACTOR"], np.nan)
    return aod_clean

# =============================================================================
# 3. 核心处理主程序
# =============================================================================
def main():
    print("=" * 80)
    print("🚀 开始构建 2025 年长三角 1km 逐日动态 AOD 特征集 ")
    print("=" * 80)
    
    # 加载目标网格基准与高程掩膜
    print("\n📌 [1/4] 加载目标网格基准与高程物理掩膜...")
    ds_target = xr.open_dataset(CONF["TARGET_GRID"])
    target_lon, target_lat = ds_target["lon"].values, ds_target["lat"].values
    elevation = ds_target["z_high"].values 
    target_height, target_width = len(target_lat), len(target_lon)
    target_shape = (target_height, target_width)
    
    target_bounds = (np.min(target_lon), np.min(target_lat), np.max(target_lon), np.max(target_lat))
    target_transform = rasterio.transform.from_bounds(*target_bounds, target_width, target_height)
    
    land_mask = elevation > 0
    
    # 扫描数据并按天分组
    print("\n📌 [2/4] 扫描 VJ119A2 文件并按日期分组...")
    all_files = glob.glob(os.path.join(CONF["DATA_DIR"], "*.h5"))
    if not all_files:
        print("❌ 未找到数据文件！")
        return
        
    date_groups = {}
    for f in all_files:
        date_str = os.path.basename(f).split(".")[1]  
        date_groups.setdefault(date_str, []).append(f)
        
    sorted_dates = sorted(list(date_groups.keys()))
    print(f"   -> 识别到 {len(sorted_dates)} 个有效观测日。")
    
    # 逐日解码、聚合及重投影
    print("\n📌 [3/4] 逐日提取 AOD、执行云掩膜并重投影对齐...")
    time_series_data = []
    valid_dates = []
    
    for date_str in tqdm(sorted_dates, desc="处理进度"):
        dest_aod_day = np.full((target_height, target_width), np.nan, dtype=np.float32)
        
        for fpath in date_groups[date_str]:
            try:
                # 检索内部子数据集路径
                with rasterio.open(fpath) as src_base:
                    aod_path = next(s for s in src_base.subdatasets if s.endswith(CONF["AOD_SDS"]))
                    qa_path = next(s for s in src_base.subdatasets if s.endswith(CONF["QA_SDS"]))
                
                with rasterio.open(aod_path) as src_aod, rasterio.open(qa_path) as src_qa:
                    # 读取多轨道立方体数据
                    aod_arr = src_aod.read().astype(np.float32)
                    qa_arr = src_qa.read()
                    nodata_val = src_aod.nodata if src_aod.nodata else -28672.0
                    
                    # 逐轨道执行质量控制
                    clean_bands = []
                    for b in range(aod_arr.shape[0]):
                        clean_band = apply_maiac_qa_mask(aod_arr[b], qa_arr[b], nodata_val)
                        clean_bands.append(clean_band)
                    
                    # 沿轨道维度聚合
                    aod_tile_2d = np.nanmean(np.array(clean_bands), axis=0)
                    
                    src_crs = src_aod.crs
                    if not src_crs: continue 
                    
                    temp_dest = np.full(target_shape, np.nan, dtype=np.float32)
                    reproject(
                        source=aod_tile_2d,
                        destination=temp_dest,
                        src_transform=src_aod.transform,
                        src_crs=src_crs,
                        dst_transform=target_transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.average, 
                        src_nodata=np.nan,
                        dst_nodata=np.nan
                    )
                    
                    # 瓦片拼图融合
                    dest_aod_day = np.fmax(dest_aod_day, temp_dest)
            except Exception:
                pass 
                
        year = int(date_str[1:5])
        day_of_year = int(date_str[5:8])
        real_date = datetime(year, 1, 1) + pd.Timedelta(days=day_of_year - 1)
        
        valid_dates.append(real_date)
        time_series_data.append(dest_aod_day)

    # 纬度维度对齐
    if target_lat[0] < target_lat[-1]:
        time_series_data = [arr[::-1, :] for arr in time_series_data]

    # Xarray 时空插值与重构
    print("\n📌 [4/4] 构建三维立方体并进行时空连续性插值 (填补云遮挡区与内陆水体)...")
    
    da_aod = xr.DataArray(
        data=np.array(time_series_data),
        dims=["time", "lat", "lon"],
        coords={"time": valid_dates, "lat": target_lat, "lon": target_lon},
        attrs={"long_name": "Aerosol Optical Depth at 0.55 um", "units": "1"}
    )
    
    # 时间维度插值
    da_daily = da_aod.interpolate_na(dim="time", method="linear")
    da_daily = da_daily.bfill(dim="time").ffill(dim="time")
    
    # 空间维度插值
    print("   -> 启动二维空间平滑修复，消灭内陆 NaN 空洞...")
    da_daily = da_daily.interpolate_na(dim="lon", method="linear")
    da_daily = da_daily.interpolate_na(dim="lat", method="linear")
    da_daily = da_daily.bfill(dim="lon").ffill(dim="lon").bfill(dim="lat").ffill(dim="lat")
    
    da_daily_2025 = da_daily.sel(time=slice("2025-01-01", "2025-12-31"))
    
    ds_final = xr.Dataset({"aod": da_daily_2025})
    
    os.makedirs(os.path.dirname(CONF["OUTPUT_NC"]), exist_ok=True)
    encoding = {"aod": {"zlib": True, "complevel": 5, "dtype": "float32", "_FillValue": np.nan}}
    ds_final.to_netcdf(CONF["OUTPUT_NC"], encoding=encoding)
    
    print("\n" + "=" * 80)
    print("✅ AOD 空间无缝修复处理完毕！")
    print(f"📊 数据维度: {ds_final['aod'].shape} (天, 纬度, 经度)")
    print(f"💾 已保存至: {CONF['OUTPUT_NC']}")

if __name__ == "__main__":
    main()