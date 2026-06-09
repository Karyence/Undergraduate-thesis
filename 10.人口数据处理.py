import xarray as xr
import numpy as np
import os
import rasterio
from rasterio.warp import reproject, Resampling
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 核心配置参数
# =============================================================================
CONF = {
    "POP_TIF": "/home/wangzonghan/Population_Raw/chn_pop_2025_CN_1km_R2025A_UA_v1.tif",
    "TARGET_GRID": os.path.expanduser("~/bisheshuju/GEBCO/GEBCO_2025_YRD_1km.nc"),
    "OUTPUT_NC": os.path.expanduser("~/bisheshuju/Population/YRD_Population_1km_2025.nc")
}

# =============================================================================
# 2. 主程序
# =============================================================================
def main():
    print("=" * 80)
    print("🚀 开始构建 2025 年长三角 1km 人口空间分布特征")
    print("=" * 80)
    
    # [步骤 1]: 加载目标 1km 基准网格
    print("\n📌 [1/3] 加载目标 1km 基准网格 (GEBCO)...")
    ds_target = xr.open_dataset(CONF["TARGET_GRID"])
    target_lon, target_lat = ds_target["lon"].values, ds_target["lat"].values
    target_height, target_width = len(target_lat), len(target_lon)
    
    # 构建目标网格的仿射变换矩阵 
    target_bounds = (np.min(target_lon), np.min(target_lat), np.max(target_lon), np.max(target_lat))
    target_transform = rasterio.transform.from_bounds(*target_bounds, target_width, target_height)
    
    # [步骤 2]: 读取并重投影人口数据
    print(f"\n📌 [2/3] 读取并提取人口数据: {os.path.basename(CONF['POP_TIF'])}")
    dest_pop = np.full((target_height, target_width), np.nan, dtype=np.float32)
    
    try:
        with rasterio.open(CONF["POP_TIF"]) as src:
            src_arr = src.read(1).astype(np.float32)
            
            # 清洗原始数据中的无效值
            nodata_val = src.nodata if src.nodata is not None else -9999
            src_arr[src_arr == nodata_val] = np.nan
            src_arr[src_arr < 0] = np.nan  # 人口不可能为负数，将异常的水体或背景剔除
            
            print("   -> 启动 面积权重分配法 (Resampling.average) 以保证人口总数守恒...")
            reproject(
                source=src_arr,
                destination=dest_pop,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.average, 
                src_nodata=np.nan,
                dst_nodata=np.nan
            )
    except Exception as e:
        print(f"❌ 读取或投影失败: {e}")
        return

    # [步骤 3]: 维度对齐与封装保存
    print("\n📌 [3/3] 维度对齐与 NetCDF 封存...")
    
    if target_lat[0] < target_lat[-1]:
        print("   -> 侦测到目标网格纬度为升序，自动执行南北翻转修正...")
        dest_pop = dest_pop[::-1, :]

    da_pop = xr.DataArray(
        dest_pop,
        dims=["lat", "lon"],
        coords={"lat": target_lat, "lon": target_lon},
        attrs={
            "long_name": "Population Count 2025",
            "units": "people per grid cell (~1km^2)",
            "source": "High-res Population Resampled via Area-Weighted Method"
        }
    )
    
    ds_output = xr.Dataset({"pop": da_pop})
    os.makedirs(os.path.dirname(CONF["OUTPUT_NC"]), exist_ok=True)
    
    encoding = {"pop": {"zlib": True, "complevel": 5, "dtype": "float32", "_FillValue": np.nan}}
    ds_output.to_netcdf(CONF["OUTPUT_NC"], encoding=encoding)
    
    # 统计计算，作为验证数据物理常识的依据
    valid_pixels = np.sum(~np.isnan(dest_pop))
    total_pop = np.nansum(dest_pop)
    
    print("\n" + "=" * 80)
    print(f"✅ 完美竣工！")
    print(f"📊 有效像元数: {valid_pixels} 个网格")
    print(f"👥 区域估算总人口: {total_pop / 10000:.2f} 万人")
    print(f"💾 已保存至: {CONF['OUTPUT_NC']}")

if __name__ == "__main__":
    main()