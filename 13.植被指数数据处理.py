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
    "DATA_DIR": "/data/Satellite/VNP13A2",
    "TARGET_GRID": os.path.expanduser("~/bisheshuju/GEBCO/GEBCO_2025_YRD_1km.nc"),
    "OUTPUT_NC": os.path.expanduser("~/bisheshuju/NDVI/YRD_NDVI_Daily_1km_2025.nc"),
    
    "NDVI_SDS": "1_km_16_days_NDVI",
    "QA_SDS": "1_km_16_days_VI_Quality",
    
    "SCALE_FACTOR": 0.0001,
    "FILL_VALUE": -3000,

    "SINU_PROJ": "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
}

# =============================================================================
# 2. 正弦投影 (Sinusoidal) 瓦片坐标变换
# =============================================================================
def get_sinu_transform(tile_h, tile_v, nx=1200, ny=1200):
    TILE_SIZE = 1111950.5196666666  # 单个瓦片物理跨度 (米)
    X_MIN_WORLD = -20015109.354     # 全球极左边界
    Y_MAX_WORLD = 10007554.677      # 全球极上边界
    
    # 计算当前瓦片的左上角坐标
    xmin = X_MIN_WORLD + tile_h * TILE_SIZE
    ymax = Y_MAX_WORLD - tile_v * TILE_SIZE
    
    # 计算空间分辨率
    pixel_size_x = TILE_SIZE / nx
    pixel_size_y = TILE_SIZE / ny
    
    return rasterio.Affine(pixel_size_x, 0.0, xmin, 0.0, -pixel_size_y, ymax)

# =============================================================================
# 3. 质量控制 (QA) 掩码解析函数
# =============================================================================
def apply_qa_mask(ndvi_arr, qa_arr):
    qa_bits_01 = qa_arr & 0b11                    # 提取 0-1 位 (整体质量)
    aerosol_bits = (qa_arr >> 6) & 0b11           # 提取 6-7 位 (气溶胶浓度)
    adj_cloud = (qa_arr >> 8) & 0b1               # 提取第 8 位 (临近云)
    mixed_cloud = (qa_arr >> 10) & 0b1            # 提取第 10 位 (混合云)
    
    # 筛选条件：质量好/一般 + 气溶胶低/中 + 无临近云 + 无混合云
    valid_mask = (qa_bits_01 <= 1) & (aerosol_bits < 3) & (adj_cloud == 0) & (mixed_cloud == 0)
    valid_mask &= (ndvi_arr != CONF["FILL_VALUE"])
    
    return np.where(valid_mask, ndvi_arr * CONF["SCALE_FACTOR"], np.nan)

# =============================================================================
# 4. 核心处理主程序
# =============================================================================
def main():
    print("=" * 80)
    print("🚀 开始构建 2025 年长三角 1km 逐日动态 NDVI 特征集 ")
    print("=" * 80)
    
    # 加载目标网格基准
    ds_target = xr.open_dataset(CONF["TARGET_GRID"])
    target_lon, target_lat = ds_target["lon"].values, ds_target["lat"].values
    target_height, target_width = len(target_lat), len(target_lon)
    target_shape = (target_height, target_width)
    
    target_bounds = (np.min(target_lon), np.min(target_lat), np.max(target_lon), np.max(target_lat))
    target_transform = rasterio.transform.from_bounds(*target_bounds, target_width, target_height)
    
    # 扫描输入数据
    all_files = glob.glob(os.path.join(CONF["DATA_DIR"], "*.h5"))
    if not all_files: return
        
    date_groups = {}
    for f in all_files:
        date_str = os.path.basename(f).split(".")[1] 
        date_groups.setdefault(date_str, []).append(f)
        
    sorted_dates = sorted(list(date_groups.keys()))
    print(f"📌 [1/3] 识别到 {len(sorted_dates)} 个时间周期。开始重投影与镶嵌...")
    
    time_series_data = []
    valid_dates = []
    
    for date_str in tqdm(sorted_dates, desc="处理进度"):
        dest_ndvi = np.full((target_height, target_width), np.nan, dtype=np.float32)
        
        for fpath in date_groups[date_str]:
            basename = os.path.basename(fpath)
            # 提取瓦片行列号
            tile_str = basename.split(".")[2]
            tile_h = int(tile_str[1:3])
            tile_v = int(tile_str[4:6])
            
            # 获取基于物理推导的绝对坐标变换矩阵
            real_transform = get_sinu_transform(tile_h, tile_v)
            
            try:
                with rasterio.open(fpath) as src_base:
                    ndvi_path = next(s for s in src_base.subdatasets if s.endswith(CONF["NDVI_SDS"]))
                    qa_path = next(s for s in src_base.subdatasets if s.endswith(CONF["QA_SDS"]))
                
                with rasterio.open(ndvi_path) as src_ndvi, rasterio.open(qa_path) as src_qa:
                    ndvi_arr = src_ndvi.read(1).astype(np.float32)
                    qa_arr = src_qa.read(1)
                    
                    ndvi_clean = apply_qa_mask(ndvi_arr, qa_arr)
                    
                    temp_dest = np.full(target_shape, np.nan, dtype=np.float32)
                    reproject(
                        source=ndvi_clean,
                        destination=temp_dest,
                        src_transform=real_transform,   # 使用推算出的绝对坐标
                        src_crs=CONF["SINU_PROJ"],      # 声明原始正弦投影
                        dst_transform=target_transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.average,
                        src_nodata=np.nan,
                        dst_nodata=np.nan
                    )
                    
                    # 瓦片拼接 (提取重叠区域的最大 NDVI 值)
                    dest_ndvi = np.fmax(dest_ndvi, temp_dest)
            except Exception:
                pass
                
        # 时间解析
        year = int(date_str[1:5])
        day_of_year = int(date_str[5:8])
        real_date = datetime(year, 1, 1) + pd.Timedelta(days=day_of_year - 1)
        
        valid_dates.append(real_date)
        time_series_data.append(dest_ndvi)

    # 纬度方向自动对齐
    if target_lat[0] < target_lat[-1]:
        time_series_data = [arr[::-1, :] for arr in time_series_data]

    print("\n📌 [2/3] 构建三维立方体并进行逐日 (Daily) 连续插值...")
    da_ndvi = xr.DataArray(
        data=np.array(time_series_data),
        dims=["time", "lat", "lon"],
        coords={"time": valid_dates, "lat": target_lat, "lon": target_lon},
    )
    
    # 时间序列插值：填补云遮挡导致的短期缺失
    da_daily = da_ndvi.resample(time="1D").interpolate("linear")
    da_daily = da_daily.bfill(dim="time").ffill(dim="time")
    
    # 物理合理性修复：水体/海洋区域赋背景值 0.0
    print("   -> 填充常年无效区 (水体/海洋)：赋背景值 0.0...")
    da_daily = da_daily.fillna(0.0)
    
    # 截取目标研究年份
    da_daily_2025 = da_daily.sel(time=slice("2025-01-01", "2025-12-31"))
    
    ds_final = xr.Dataset({"ndvi": da_daily_2025})
    
    os.makedirs(os.path.dirname(CONF["OUTPUT_NC"]), exist_ok=True)
    ds_final.to_netcdf(CONF["OUTPUT_NC"], encoding={"ndvi": {"zlib": True, "complevel": 5, "dtype": "float32", "_FillValue": np.nan}})
    
    print(f"\n✅ [3/3] 完美竣工！数据已保存至: {CONF['OUTPUT_NC']}")

if __name__ == "__main__":
    main()