import os
import xarray as xr
import chinese_calendar as conc
import warnings
import joblib
import gc
import cudf
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 配置 
# =============================================================================
MAX_CPU_WORKERS = 8 
BASE_DIR = "/home/wangzonghan/bisheshuju"
OUTPUT_DIR = "/data/Machine_Learning/wangzonghan/YRD_PM25_Project/Inference_Features"
MODEL_DIR = f"{BASE_DIR}/Results/Models_随机森林"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    'AOD': f"{BASE_DIR}/AOD/YRD_AOD_Daily_1km_2025.nc",
    'NDVI': f"{BASE_DIR}/NDVI/YRD_NDVI_Daily_1km_2025.nc",
    'POPULATION': f"{BASE_DIR}/Population/YRD_Population_1km_2025.nc",
    'LC': f"{BASE_DIR}/LandCover/YRD_LandCover_Fractions_1km_Final.nc",
    'BLH': f"{BASE_DIR}/ERA5_LAND/ERA5_BLH_1km_downscaled.nc",
    'D2M': f"{BASE_DIR}/ERA5_LAND/ERA5_d2m_1km_downscaled.nc",
    'T2M': f"{BASE_DIR}/ERA5_LAND/ERA5_t2m_1km_downscaled.nc",
    'TP': f"{BASE_DIR}/ERA5_LAND/ERA5_tp_1km_downscaled.nc",
    'WIND': f"{BASE_DIR}/ERA5_LAND/ERA5_wind_speed_1km_downscaled.nc",
    'TERRAIN': f"{BASE_DIR}/Grid_Features/Grid_Terrain_Features.parquet"
}

# =============================================================================
# 2. 核心算法工场 
# =============================================================================
def apply_advanced_gap_filling_gpu(gdf):
    # 1. 静态特征修复 
    for col in ['DEM', 'Slope', 'POPULATION', 'NDVI']:
        if col in gdf.columns: gdf[col] = gdf[col].fillna(0)
    
    # 2. 土地利用修复
    lc_cols = [c for c in gdf.columns if c.startswith('LC_')]
    if 'LC_water_frac' in lc_cols:
        gdf['LC_water_frac'] = gdf['LC_water_frac'].fillna(1.0)
        other_lc = [c for c in lc_cols if c != 'LC_water_frac']
        gdf[other_lc] = gdf[other_lc].fillna(0.0)
    
    # 3. 动态特征修复 
    dyn_cols = ['AOD', 'ERA5_BLH', 'ERA5_D2M', 'ERA5_T2M', 'ERA5_TP', 'ERA5_WIND']
    existing = [c for c in dyn_cols if c in gdf.columns]
    
    # 按空间和时间排序以进行时序填充 
    gdf = gdf.sort_values(by=['lat', 'lon', 'hour'])
    
    for col in existing:
        gdf[col] = gdf[col].ffill().bfill()
        col_median = gdf[col].median()
        if not np.isnan(col_median):
            gdf[col] = gdf[col].fillna(col_median)
        else:
            gdf[col] = gdf[col].fillna(0)
            
    return gdf

def read_daily_data(target_date):
    try:
        results = {'daily': {}, 'hourly': {}}
        vars_to_read = ['AOD', 'NDVI', 'BLH', 'D2M', 'T2M', 'TP', 'WIND']
        for var in vars_to_read:
            with xr.open_dataset(FILES[var]) as ds:
                time_dim = 'time' if 'time' in ds.dims else list(ds.dims.keys())[0]
                if var in ['BLH', 'D2M', 'T2M', 'TP', 'WIND']:
                    df = ds.sel({time_dim: target_date}).to_dataframe().reset_index()
                    df['hour'] = df[time_dim].dt.hour.astype('int32')
                    name = f"ERA5_{var}"
                else:
                    df = ds.sel({time_dim: target_date}, method='nearest').to_dataframe().reset_index()
                    name = var
                
                df = df.rename(columns={'longitude':'lon', 'latitude':'lat'})
                val_cols = [c for c in df.columns if c not in ['lon','lat','time','date','hour','spatial_ref','x','y']]
                df = df.rename(columns={val_cols[0]: name})
                df['lon'], df['lat'] = df['lon'].round(3), df['lat'].round(3)
                
                if 'hour' in df.columns:
                    results['hourly'][name] = df[['lon', 'lat', 'hour', name]]
                else:
                    results['daily'][name] = df[['lon', 'lat', name]]
        return target_date, results
    except Exception as e:
        return target_date, str(e)

# =============================================================================
# 3. 主进程驱动
# =============================================================================
def main():
    expected_features = joblib.load(os.path.join(MODEL_DIR, "best_features_list_terrain.pkl"))
    
    # --- A. 静态底座预构建 ---
    print("🌍 正在构建 1700万行 高精度静态底座并卸载至内存...")
    static_gdf = cudf.read_parquet(FILES['TERRAIN'])[['lon', 'lat', 'DEM', 'Slope']]
    static_gdf['lon'], static_gdf['lat'] = static_gdf['lon'].round(3), static_gdf['lat'].round(3)

    with xr.open_dataset(FILES['POPULATION']) as ds:
        pop_df = ds.to_dataframe().reset_index().rename(columns={'longitude':'lon','latitude':'lat'})
        pop_gdf = cudf.from_pandas(pop_df[['lon', 'lat', pop_df.columns[-1]]].rename(columns={pop_df.columns[-1]:'POPULATION'}))
        pop_gdf['lon'], pop_gdf['lat'] = pop_gdf['lon'].round(3), pop_gdf['lat'].round(3)

    with xr.open_dataset(FILES['LC']) as ds:
        lc_df = ds.to_dataframe().reset_index().rename(columns={'longitude':'lon','latitude':'lat'})
        lc_raw_cols = [c for c in lc_df.columns if c not in ['lon','lat','time','date','spatial_ref','x','y']]
        lc_df = lc_df.rename(columns={c: f"LC_{c}" for c in lc_raw_cols})
        lc_gdf = cudf.from_pandas(lc_df[['lon', 'lat'] + [f"LC_{c}" for c in lc_raw_cols]])
        lc_gdf['lon'], lc_gdf['lat'] = lc_gdf['lon'].round(3), lc_gdf['lat'].round(3)

    static_base = static_gdf.merge(pop_gdf, on=['lon', 'lat'], how='outer')
    static_base = static_base.merge(lc_gdf, on=['lon', 'lat'], how='outer')
    
    hours_gdf = cudf.DataFrame({'hour': np.arange(24, dtype='int32')})
    # 显存关键点：在合并后立刻转为 Pandas 移出显存
    static_base_24h_cpu = static_base.merge(hours_gdf, how='cross').to_pandas()
    
    del static_gdf, pop_gdf, lc_gdf, static_base, hours_gdf
    gc.collect()

    # --- B. 任务分配 ---
    date_list = [d.strftime('%Y-%m-%d') for d in pd.date_range('2025-01-01', '2025-12-31')]
    print(f"🔥 高精度+显存优化模式启动 | 核心数: {MAX_CPU_WORKERS}")

    with ProcessPoolExecutor(max_workers=MAX_CPU_WORKERS) as executor:
        future_data = executor.map(read_daily_data, date_list)

        for target_date, data in future_data:
            if isinstance(data, str):
                print(f"❌ {target_date} 错误: {data}"); continue
            
            print(f"🚀 GPU 正在处理 [{target_date}] ...")
            
            # 每轮循环开始时才把底座送入 GPU
            final_gdf = cudf.from_pandas(static_base_24h_cpu)
            
            # 挂载动态数据
            for name, v in data['daily'].items():
                final_gdf = final_gdf.merge(cudf.from_pandas(v), on=['lon', 'lat'], how='left')
            for name, v in data['hourly'].items():
                final_gdf = final_gdf.merge(cudf.from_pandas(v), on=['lon', 'lat', 'hour'], how='left')

            # 时间标签
            dt_obj = datetime.strptime(target_date, '%Y-%m-%d')
            final_gdf['month'] = dt_obj.month
            final_gdf['season'] = (dt_obj.month % 12 // 3 + 1)
            final_gdf['day_of_week'] = dt_obj.weekday()
            final_gdf['is_holiday'] = 1 if conc.is_holiday(dt_obj) else 0
            final_gdf['is_weekend'] = 1 if dt_obj.weekday() in [5, 6] else 0
            
            # 高精度物理填充
            final_gdf = apply_advanced_gap_filling_gpu(final_gdf)

            # --- C. 最终输出与体检 ---
            for feat in expected_features:
                if feat not in final_gdf.columns: final_gdf[feat] = 0.0
            
            final_output = final_gdf[['lon', 'lat'] + expected_features].astype('float32')

            # 🚨 数据健康检查
            for col in final_output.columns:
                if np.isinf(final_output[col].max()) or np.isinf(final_output[col].min()):
                    print(f"🚨 警告: [{col}] 包含 Inf，正在使用中位数修复...")
                    m = final_output[col].replace([np.inf, -np.inf], np.nan).median()
                    final_output[col] = final_output[col].replace([np.inf, -np.inf], m if not np.isnan(m) else 0)

            # 强制执行显存碎片整理
            final_output = final_output.copy() 
            gc.collect()

            try:
                output_file = f"{OUTPUT_DIR}/YRD_Grid_Hourly_Features_Terrain_{target_date.replace('-','')}.parquet"
                # 写入 Parquet
                final_output.to_parquet(output_file, compression='zstd', index=False)
                print(f"✅ [{target_date}] 成功写入 /data 盘")
            except Exception as e:
                print(f"🔥 写入崩溃: {str(e)}"); raise e 
            
            # 💥 彻底清空本轮显存
            del final_gdf, final_output
            gc.collect()

    print("🎉 全年高精度特征库构建完成！")

if __name__ == "__main__":
    main()