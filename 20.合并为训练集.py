import os
import pandas as pd
import xarray as xr
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 核心路径配置字典
# =============================================================================
PATHS = {
    "LABEL": "/home/wangzonghan/bisheshuju/PM25_Labels/YRD_PM25_Hourly_2025_Sites.csv",
    
    # 动态特征 3D
    "AOD": "/home/wangzonghan/bisheshuju/AOD/YRD_AOD_Daily_1km_2025.nc",
    "NDVI": "/home/wangzonghan/bisheshuju/NDVI/YRD_NDVI_Daily_1km_2025.nc",
    "ERA5_BLH": "/home/wangzonghan/bisheshuju/ERA5_LAND/ERA5_BLH_1km_downscaled.nc",
    "ERA5_D2M": "/home/wangzonghan/bisheshuju/ERA5_LAND/ERA5_d2m_1km_downscaled.nc",
    "ERA5_T2M": "/home/wangzonghan/bisheshuju/ERA5_LAND/ERA5_t2m_1km_downscaled.nc",
    "ERA5_TP": "/home/wangzonghan/bisheshuju/ERA5_LAND/ERA5_tp_1km_downscaled.nc",
    "ERA5_WIND": "/home/wangzonghan/bisheshuju/ERA5_LAND/ERA5_wind_speed_1km_downscaled.nc",
    
    # 静态特征 2D
    "GEBCO": "/home/wangzonghan/bisheshuju/GEBCO/GEBCO_2025_YRD_1km_with_slope.nc",
    "POP": "/home/wangzonghan/bisheshuju/Population/YRD_Population_1km_2025.nc",
    "LANDCOVER": "/home/wangzonghan/bisheshuju/LandCover/YRD_LandCover_Fractions_1km_Final.nc",
    
    # 最终输出
    "OUTPUT_PARQUET": "/home/wangzonghan/bisheshuju/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet",
    "OUTPUT_CSV": "/home/wangzonghan/bisheshuju/训练集/YRD_PM25_Hourly_ML_Dataset_2025.csv" 
}

def get_main_var(ds):
    vars_list = list(ds.data_vars)
    valid_vars = [v for v in vars_list if v not in ['spatial_ref', 'crs', 'transverse_mercator']]
    return valid_vars[0] if valid_vars else None

# =============================================================================
# 🚀 极速一维最近邻索引查找
# =============================================================================
def fast_nearest_idx(array, values):
    """摒弃 xarray 内部慢查询，使用纯 numpy 二分查找实现光速匹配"""
    array = np.asarray(array)
    
    # 自动处理时间与浮点数类型
    if np.issubdtype(array.dtype, np.datetime64):
        array = array.astype('datetime64[s]').astype(np.int64)
        values = np.asarray(values).astype('datetime64[s]').astype(np.int64)
    else:
        array = array.astype(np.float64)
        values = np.asarray(values).astype(np.float64)
        
    is_descending = array[0] > array[-1]
    if is_descending:
        array = array[::-1]
        
    idx = np.searchsorted(array, values, side="left")
    idx = np.clip(idx, 1, len(array)-1)
    
    left = array[idx-1]
    right = array[idx]
    idx -= (values - left) < (right - values)
    
    if is_descending:
        idx = len(array) - 1 - idx
    return idx

# =============================================================================
# 主程序：光速时空扁平化与对齐合并
# =============================================================================
def build_dataset():
    print("=" * 80)
    print("🚀 启动 PM2.5 机器学习特征融合 ")
    print("=" * 80)

    # [步骤 1]: 加载标签
    print("\n📌 [1/3] 加载地面真实标签 (Y)...")
    df = pd.read_csv(PATHS["LABEL"])
    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    
    pts_lat = df['lat'].values
    pts_lon = df['lon'].values
    pts_datetime = df['datetime'].values
    
    initial_len = len(df)
    print(f"   -> 成功加载 {initial_len:,} 个目标时空样本点。")

    # [步骤 2]: 遍历所有特征，光速提取
    print("\n📌 [2/3] 开启特征提取...")
    
    all_keys = ["AOD", "NDVI", "ERA5_BLH", "ERA5_D2M", "ERA5_T2M", "ERA5_TP", "ERA5_WIND", "GEBCO", "POP", "LANDCOVER"]
    
    for key in all_keys:
        path = PATHS[key]
        if not os.path.exists(path):
            print(f"   ⚠️ 找不到 {key} 文件，跳过。")
            continue
            
        try:
            print(f"   ⏳ 正在将 {key} 读取到内存...", end="", flush=True)
            ds = xr.open_dataset(path).load()
            print(" [读取完成，瞬间提取中...]")
            
            # 光速计算经纬度索引
            lat_idx = fast_nearest_idx(ds['lat'].values, pts_lat)
            lon_idx = fast_nearest_idx(ds['lon'].values, pts_lon)
            
            # 区分处理静态/动态变量
            if key in ["GEBCO", "POP", "LANDCOVER"]:
                if key == "GEBCO":
                    dem_var = 'DEM' if 'DEM' in ds else 'dem'
                    slope_var = 'Slope' if 'Slope' in ds else 'slope'
                    df['DEM'] = ds[dem_var].values[lat_idx, lon_idx]
                    df['Slope'] = ds[slope_var].values[lat_idx, lon_idx]
                elif key == "POP":
                    df['POPULATION'] = ds[get_main_var(ds)].values[lat_idx, lon_idx]
                elif key == "LANDCOVER":
                    lc_vars = [v for v in ds.data_vars if v not in ['spatial_ref', 'crs']]
                    for lc_var in lc_vars:
                        df[f'LC_{lc_var}'] = ds[lc_var].values[lat_idx, lon_idx]
            else:
                var_name = get_main_var(ds)
                
                # 时区纠正处理
                if "ERA5" in key:
                    pts_time_query = pts_datetime - np.timedelta64(8, 'h')
                else:
                    pts_time_query = pts_datetime
                    
                time_idx = fast_nearest_idx(ds['time'].values, pts_time_query)
                df[key] = ds[var_name].values[time_idx, lat_idx, lon_idx]
                
            print(f"   ✅ {key} 提取成功！")
            ds.close()
            
        except Exception as e:
            print(f"\n   ❌ {key} 提取失败: {e}")

    # [步骤 3]: 终极清洗与输出
    print("\n📌 [3/3] 终极清洗与输出生成...")
    print("\n" + "-"*60)
    print("🕵️‍♂️ 侦探模式：全网格 NaN 溯源与站点存活分析")
    
    # 1. 打印有缺失值的特征列及缺失数量
    print("\n[A] 各列缺失值 (NaN) 统计 (只显示有缺失的列):")
    missing_stats = df.isna().sum()
    print(missing_stats[missing_stats > 0])
    
    # 2. 统计清洗前后的站点集合
    sites_before = set(df['site_code'].unique())
    df_clean = df.dropna().reset_index(drop=True)
    sites_after = set(df_clean['site_code'].unique())
    
    # 3. 找出被彻底剔除的站点
    dropped_sites = sites_before - sites_after
    
    print(f"\n[B] 站点存活分析:")
    print(f"   -> 原始输入站点数: {len(sites_before)}")
    print(f"   -> 最终存活站点数: {len(sites_after)}")
    print(f"   -> 彻底被剔除的站点数: {len(dropped_sites)}")
    
    if len(dropped_sites) > 0:
        print("\n[C] 结果报告 (这批被剔除的站点，到底缺了什么特征):")
        df_dropped_sites = df[df['site_code'].isin(dropped_sites)]
        dropped_missing_stats = df_dropped_sites.isna().sum()
        print(dropped_missing_stats[dropped_missing_stats > 0])
    print("-" * 60 + "\n")
    
    final_len = len(df_clean)
    print(f"   -> 清洗前样本数: {initial_len:,}")
    print(f"   -> 清洗后样本数: {final_len:,} (自动剔除了 {initial_len - final_len:,} 个包含 NaN 的记录)")
    
    # 导出
    df_clean.to_csv(PATHS["OUTPUT_CSV"], index=False)
    df_clean.to_parquet(PATHS["OUTPUT_PARQUET"], engine='pyarrow', index=False, compression='snappy')
    
    print("\n" + "=" * 80)
    print("🏆 伟大竣工！你的【百万级小时距】机器学习 DataFrame 已经铸造完成！")
    print(f"📊 特征矩阵维度: {df_clean.shape[0]:,} 行 × {df_clean.shape[1]} 列")
    print(f"💡 注意：表内已成功保留 'hour' 作为独立特征喂给模型！")
    print(f"💾 Parquet 训练集已极速保存至: {PATHS['OUTPUT_PARQUET']}")
    print("=" * 80)

if __name__ == "__main__":
    build_dataset()