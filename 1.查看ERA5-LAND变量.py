import xarray as xr
import os
import numpy as np

# =============================================================================
# 配置参数
# =============================================================================
ERA5_FILE = "/data/Climate/ERA5/ERA5-LAND/era_land_20250101.nc" 
KEY_VARS = ["t2m", "u10", "v10", "d2m", "sp", "tp"] 

# =============================================================================
# 核心函数：查看单个 ERA5 文件变量信息
# =============================================================================
def check_era5_single_file_vars(file_path):
    print("="*70)
    print(f"📊 ERA5 单文件变量查看（文件：{os.path.basename(file_path)}）")
    print("="*70)

    # 1. 加载文件
    try:
        ds = xr.open_dataset(
            file_path,
            engine='netcdf4',
            decode_times=True,
            chunks={'valid_time': 1} 
        )
        print(f"数据加载成功！文件格式：{ds.attrs.get('Conventions', 'CF-1.6')}")
    except Exception as e:
        print(f"数据加载失败：{str(e)[:100]}...")
        return

    # 2. 自动识别时间维度
    time_dims = [dim for dim in ds.dims if 'time' in dim.lower()]
    if not time_dims:
        print(f"未识别到时间维度！文件维度：{dict(ds.dims)}")
        time_dim = None
    else:
        time_dim = time_dims[0]
        print(f"识别时间维度：{time_dim}（共 {ds.dims[time_dim]} 个时间步）")

    # 3. 输出文件整体结构
    print(f"\n📋 文件整体结构")
    print(f"   数据维度：{dict(ds.dims)}")
    print(f"   所有变量（共 {len(ds.data_vars)} 个）：{list(ds.data_vars)}")
    
    if time_dim:
        try:
            time_min = ds[time_dim].min().values
            time_max = ds[time_dim].max().values
            print(f"   时间范围：{time_min} 至 {time_max}")
        except Exception as e:
            print(f"   时间范围解析失败：{str(e)[:50]}...")

    # 4. 重点变量详细信息
    print(f"\n🎯 核心变量详细信息")
    found_key_vars = [var for var in KEY_VARS if var in ds.data_vars]
    missing_key_vars = [var for var in KEY_VARS if var not in ds.data_vars]

    if found_key_vars:
        for var_name in found_key_vars:
            da = ds[var_name]
            var_dims = da.dims
            print(f"\n   【变量：{var_name}】")
            print(f"   - 变量全称：{da.attrs.get('long_name', '未定义')}")
            print(f"   - 单位：{da.attrs.get('units', '未定义')}")
            print(f"   - 数据维度：{da.shape}（{var_dims}）")
            print(f"   - 数据类型：{da.dtype} | 填充值：{da.attrs.get('_FillValue', '无')}")
            
            if da.attrs.get('_FillValue') is not None:
                valid_data = da.where(da != da.attrs['_FillValue'])
            else:
                valid_data = da
            if valid_data.size > 0:
                print(f"   - 数值范围：{np.nanmin(valid_data.values):.2f} ~ {np.nanmax(valid_data.values):.2f}")
    else:
        print(f"   未找到核心变量！")

    if missing_key_vars:
        print(f"\n   缺失核心变量：{missing_key_vars}")

    # 5. 数据可用性结论
    print(f"\n" + "="*70)
    if len(found_key_vars) >= 3: 
        print("数据完整度评估：满足后续降尺度与处理需求")
    else:
        print("数据完整度评估：核心变量不足")
    print("="*70)

    ds.close()

# =============================================================================
# 主控入口
# =============================================================================
if __name__ == "__main__":
    check_era5_single_file_vars(ERA5_FILE)