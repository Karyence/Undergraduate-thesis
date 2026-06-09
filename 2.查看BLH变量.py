import xarray as xr
import os
import pandas as pd

# =============================================================================
# 配置参数
# =============================================================================
INPUT_FILE = "/home/wangzonghan/ERA5-BLH/BLH.nc" 

# =============================================================================
# 核心查看逻辑
# =============================================================================
def check_blh_nc():
    print("="*60)
    print(f"BLH 数据文件检查: {os.path.basename(INPUT_FILE)}")
    print("="*60)

    try:
        ds = xr.open_dataset(INPUT_FILE, engine='netcdf4')
        
        print("\n[文件整体概况]")
        print(f"维度: {dict(ds.dims)}")
        print(f"包含变量: {list(ds.data_vars)}")
        
        if 'longitude' in ds.coords and 'latitude' in ds.coords:
            lon = ds['longitude']
            lat = ds['latitude']
            print(f"空间范围: 经度 {lon.min().item():.2f}~{lon.max().item():.2f}, 纬度 {lat.min().item():.2f}~{lat.max().item():.2f}")
            print(f"空间分辨率: 经度步长 ≈ {abs(lon[1]-lon[0]).item():.4f}°, 纬度步长 ≈ {abs(lat[1]-lat[0]).item():.4f}°")
        
        time_dim = None
        for t in ['time', 'valid_time']:
            if t in ds.coords:
                time_dim = t
                break
        
        if time_dim:
            time_vals = pd.to_datetime(ds[time_dim].values)
            print(f"时间范围: {time_vals.min()} 至 {time_vals.max()}")
            print(f"时间步数: {len(time_vals)}")
        else:
            print("未检测到标准时间维度 (time/valid_time)")

        print("\n[变量详细属性]")
        target_vars = ['blh', 'boundary_layer_height'] 
        found_vars = [v for v in ds.data_vars if v in target_vars]
        
        if not found_vars:
            found_vars = list(ds.data_vars)
            
        for var_name in found_vars:
            da = ds[var_name]
            print(f"变量名: {var_name}")
            print(f"  - 全名: {da.attrs.get('long_name', 'N/A')}")
            print(f"  - 单位: {da.attrs.get('units', 'N/A')}")
            print(f"  - 形状: {da.shape}")
            print(f"  - 数据类型: {da.dtype}")
            
            try:
                sample_data = da.isel({time_dim: 0}) if time_dim and time_dim in da.dims else da
                v_min = sample_data.min().item()
                v_max = sample_data.max().item()
                v_mean = sample_data.mean().item()
                print(f"  - 首帧统计 (Min/Max/Mean): {v_min:.2f} / {v_max:.2f} / {v_mean:.2f}")
            except Exception:
                pass
            print("-" * 30)

        ds.close()

    except Exception as e:
        print(f"读取失败: {e}")

if __name__ == "__main__":
    check_blh_nc()