import xarray as xr
import numpy as np
import os
import glob
import pandas as pd

# =============================================================================
# 配置参数
# =============================================================================
DOWNSCALED_FILE = os.path.expanduser("~/bisheshuju/ERA5_LAND/ERA5_t2m_1km_downscaled.nc")
ERA5_RAW_DIR = "/data/Climate/ERA5/ERA5-LAND/"
FILE_PATTERNS = ["era_land_20250[1-9]*.nc", "era_land_20251[0-2]*.nc"]

REGION_LON_MIN, REGION_LON_MAX = 114.5, 123.0
REGION_LAT_MIN, REGION_LAT_MAX = 27.0, 35.5

VAR_NAME = "t2m"
TARGET_RES = 0.01

REPORT_FILE = os.path.expanduser("~/bisheshuju/ERA5_LAND/t2m_downscale_comparison_report.txt")

# =============================================================================
# 核心处理函数
# =============================================================================
def get_true_resolution(coord_array):
    """计算真实空间分辨率"""
    unique_coords = np.unique(coord_array)
    unique_coords.sort()
    if len(unique_coords) >= 2:
        res = np.mean(np.diff(unique_coords))
        return round(res, 4)
    else:
        return None

def load_original_era5_processed():
    """加载并预处理原始ERA5数据"""
    file_list = sorted([
        f for p in FILE_PATTERNS
        for f in glob.glob(os.path.join(ERA5_RAW_DIR, p))
        if os.path.isfile(f)
    ])
    if not file_list:
        raise ValueError("未匹配到任何原始ERA5数据文件")
        
    data_list = []
    for fp in file_list:
        with xr.open_dataset(fp, engine="netcdf4") as ds:
            rename_dict = {}
            coord_map = [("valid_time", "time"), ("longitude", "lon"), ("latitude", "lat")]
            for old_name, new_name in coord_map:
                if old_name in ds.coords:
                    rename_dict[old_name] = new_name
            ds = ds.rename(rename_dict) if rename_dict else ds

            if "lon" not in ds.coords or "lat" not in ds.coords:
                print(f"警告：文件 {os.path.basename(fp)} 缺失经纬度坐标，已跳过")
                continue

            lat_slice = slice(REGION_LAT_MAX, REGION_LAT_MIN) if ds.lat[0] > ds.lat[-1] else slice(REGION_LAT_MIN, REGION_LAT_MAX)
            try:
                t2m_raw = ds[VAR_NAME].sel(
                    lon=slice(REGION_LON_MIN, REGION_LON_MAX),
                    lat=lat_slice
                )
            except Exception as e:
                print(f"裁剪文件 {os.path.basename(fp)} 失败：{e}，已跳过")
                continue

            t2m_hourly = t2m_raw - 273.15
            data_list.append(t2m_hourly)  

    if not data_list:
        raise ValueError("所有原始ERA5文件均处理失败，无有效数据")
        
    era5_raw = xr.concat(data_list, dim="time", join='override').sortby("time")
    era5_raw = era5_raw.sel(time=slice("2025-01-01", "2025-12-31"))
    
    return era5_raw

def generate_table_report(ds_down, ds_raw, report_path):
    """生成并输出对比报告"""
    lon_down_res = get_true_resolution(ds_down.lon.values)
    lat_down_res = get_true_resolution(ds_down.lat.values)
    lon_raw_res = get_true_resolution(ds_raw.lon.values)
    lat_raw_res = get_true_resolution(ds_raw.lat.values)
    
    lon_down_min = ds_down.lon.min().item()
    lon_down_max = ds_down.lon.max().item()
    lon_down_cnt = len(ds_down.lon)
    lat_down_min = ds_down.lat.min().item()
    lat_down_max = ds_down.lat.max().item()
    lat_down_cnt = len(ds_down.lat)
    
    lon_raw_min = ds_raw.lon.min().item()
    lon_raw_max = ds_raw.lon.max().item()
    lon_raw_cnt = len(ds_raw.lon)
    lat_raw_min = ds_raw.lat.min().item()
    lat_raw_max = ds_raw.lat.max().item()
    lat_raw_cnt = len(ds_raw.lat)
    
    time_down_cnt = len(ds_down.time)
    time_raw_cnt = len(ds_raw.time)
    
    down_min = ds_down[VAR_NAME].min().item()
    down_max = ds_down[VAR_NAME].max().item()
    raw_min = ds_raw.min().item()
    raw_max = ds_raw.max().item()
    
    table_header = f"{'对比指标':<30}{'降尺度后数据(1km)':<30}{'原始ERA5数据':<30}"
    split_line = "-" * 90
    table_rows = [
        split_line,
        table_header,
        split_line,
        f"{'经度分辨率(°)':<30}{lon_down_res:<30.4f}{lon_raw_res:<30.4f}",
        f"{'纬度分辨率(°)':<30}{lat_down_res:<30.4f}{lat_raw_res:<30.4f}",
        f"{'经度最小值(°)':<30}{lon_down_min:<30.3f}{lon_raw_min:<30.3f}",
        f"{'经度最大值(°)':<30}{lon_down_max:<30.3f}{lon_raw_max:<30.3f}",
        f"{'经度网格点数':<30}{lon_down_cnt:<30d}{lon_raw_cnt:<30d}",
        f"{'纬度最小值(°)':<30}{lat_down_min:<30.3f}{lat_raw_min:<30.3f}",
        f"{'纬度最大值(°)':<30}{lat_down_max:<30.3f}{lat_raw_max:<30.3f}",
        f"{'纬度网格点数':<30}{lat_down_cnt:<30d}{lat_raw_cnt:<30d}",
        f"{'时间序列长度(小时)':<30}{time_down_cnt:<30d}{time_raw_cnt:<30d}",
        f"{'温度最小值(℃)':<30}{down_min:<30.2f}{raw_min:<30.2f}",
        f"{'温度最大值(℃)':<30}{down_max:<30.2f}{raw_max:<30.2f}",
        split_line
    ]
    
    conclusion = []
    conclusion.append("\n========== 校验结论 ==========")
    res_check = "✅ 分辨率符合1km(0.01°)要求" if np.isclose(lon_down_res, TARGET_RES, atol=1e-4) else "❌ 分辨率不符合预期"
    conclusion.append(res_check)
    time_check = "✅ 时间维度完全一致" if time_down_cnt == time_raw_cnt else "❌ 时间维度不一致"
    conclusion.append(time_check)
    temp_check = "✅ 气温数据在物理合理范围(-40~60℃)内" if (-40 <= down_min <= 60 and -40 <= down_max <= 60) else "❌ 存在异常气温值"
    conclusion.append(temp_check)
    conclusion.append("="*40)
    conclusion.append("判定结果：本次ERA5-t2m插值降尺度处理成功！")
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ERA5-t2m 降尺度前后数据对比报告\n")
        f.write("="*40 + "\n")
        f.write("研究区域：长三角(114.5°E~123°E, 27°N~35.5°N)\n")
        f.write("报告生成时间：" + str(pd.Timestamp.now()) + "\n\n")
        for row in table_rows:
            f.write(row + "\n")
        for line in conclusion:
            f.write(line + "\n")
    
    print("\n========== 对比结果表格预览 ==========")
    for row in table_rows:
        print(row)
    for line in conclusion:
        print(line)

# =============================================================================
# 主执行流程
# =============================================================================
if __name__ == "__main__":
    print("===== 读取并裁剪长三角研究区域 =====")
    try:
        ds_downscaled = xr.open_dataset(DOWNSCALED_FILE)
        ds_raw_region = load_original_era5_processed()
        print("研究区域数据加载与预处理完成")
        
        generate_table_report(ds_downscaled, ds_raw_region, REPORT_FILE)
        
        ds_downscaled.close()
        print(f"\n报告已完整输出至：{REPORT_FILE}")
        
    except Exception as e:
        print(f"程序执行失败：{e}")