import os
import xarray as xr
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 核心配置参数
# =============================================================================
CONF = {
    # 输入: 原始 1km GEBCO 数据 
    "GEBCO_1KM_NC": os.path.expanduser("~/bisheshuju/GEBCO/GEBCO_2025_YRD_1km.nc"),
    
    # 输出 1: 带有 DEM 和 Slope 两个波段的全新 .nc 地图文件
    "OUT_NC_WITH_SLOPE": "/home/wangzonghan/bisheshuju/GEBCO/GEBCO_2025_YRD_1km_with_slope.nc",
    
    # 输出 2: 全空间 1km 网格地形特征表 (供最终全境空间反演预测使用)
    "OUT_GRID_TERRAIN": "/home/wangzonghan/bisheshuju/Grid_Features/Grid_Terrain_Features.parquet"
}

# =============================================================================
# 2. 地理计算辅助函数
# =============================================================================
def calculate_slope(dem_array, res_degrees=0.01):
    """基于高程矩阵计算地表坡度 (单位: 度)"""
    # 依据中纬度(约31°N)计算物理距离近似值
    dx_meters = res_degrees * 111320 * np.cos(np.radians(31.0)) 
    dy_meters = res_degrees * 111320                            

    # 计算XY方向高程梯度
    dy, dx = np.gradient(dem_array, dy_meters, dx_meters)
    
    # 将梯度转换为坡度角
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    
    return slope_deg

# =============================================================================
# 3. 核心处理主程序
# =============================================================================
def main():
    print("="*70)
    print("⛰️ [1/3] 正在读取并计算长三角全空间地形特征 (DEM & Slope)...")
    print("="*70)
    
    # 读取目标网格数据
    if not os.path.exists(CONF["GEBCO_1KM_NC"]):
        print(f"❌ 找不到高程网格数据: {CONF['GEBCO_1KM_NC']}")
        return
        
    ds = xr.open_dataset(CONF["GEBCO_1KM_NC"])
    
    # 识别真实的物理变量名称并提取矩阵
    valid_vars = [v for v in list(ds.data_vars) if v not in ['spatial_ref', 'crs', 'transverse_mercator']]
    orig_var = valid_vars[0] 
    dem_array = np.squeeze(ds[orig_var].values)
    
    print(f"  -> 🎯 成功绕开伪变量，抓取到真实地形波段: [{orig_var}]")
    print(f"  -> 🗺️ 确认高程矩阵维度: {dem_array.shape}")
    
    lons = ds.lon.values
    lats = ds.lat.values
    
    # 执行坡度计算与物理掩膜(清理海洋区域)
    print(" ⏳ 正在计算坡度矩阵...")
    slope_array = calculate_slope(dem_array, res_degrees=0.01)
    
    dem_array = np.where(dem_array < 0, 0, dem_array)
    slope_array = np.where(dem_array == 0, 0, slope_array)
    
    print("\n" + "="*70)
    print("🗺️ [2/3] 正在生成全新的多波段 .nc 地形文件...")
    print("="*70)
    
    ds['DEM'] = (('lat', 'lon'), dem_array)
    ds['Slope'] = (('lat', 'lon'), slope_array)
    
    # 移除冗余单变量，保持数据结构纯净
    if orig_var not in ['DEM', 'Slope']:
        ds = ds.drop_vars(orig_var)
        
    ds.to_netcdf(CONF["OUT_NC_WITH_SLOPE"])
    print(f" ✅ 带有双地形特征的 .nc 地图已成功生成: {CONF['OUT_NC_WITH_SLOPE']}")
    
    print("\n" + "="*70)
    print("🎯 [3/3] 正在输出 Flatten 全空间网格地形表...")
    print("="*70)
    
    # 构建全空间展平网格表 (Grid Table)
    print(" ⏳ 正在打包全空间网格特征 (Parquet)...")
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid_df = pd.DataFrame({
        'lon': lon_grid.ravel(),
        'lat': lat_grid.ravel(),
        'DEM': dem_array.ravel(),
        'Slope': slope_array.ravel()
    })
    
    # 清理掉海洋(NaN或负值引发的冗余)
    grid_df = grid_df.dropna()
    grid_df.to_parquet(CONF["OUT_GRID_TERRAIN"], index=False)
    
    print(f" ✅ 全空间地形网格表已保存至: {CONF['OUT_GRID_TERRAIN']}")

if __name__ == "__main__":
    main()