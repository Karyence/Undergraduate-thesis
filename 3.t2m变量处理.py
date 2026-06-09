import xarray as xr
import numpy as np
import os
import glob
from dask.diagnostics import ProgressBar
import warnings
from dask.distributed import Client, LocalCluster

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 配置参数
# =============================================================================
ERA5_DIR = "/data/Climate/ERA5/ERA5-LAND/"
DEM_PATH = os.path.expanduser("~/bisheshuju/GEBCO/GEBCO_2025_YRD_1km.nc")
OUTPUT_PATH = os.path.expanduser("~/bisheshuju/ERA5_LAND/ERA5_t2m_1km_downscaled.nc")

FILE_PATTERNS = ["era_land_20250[1-9]*.nc", "era_land_20251[0-2]*.nc"]

CHUNKS_ERA5 = {"time": 100}  
CHUNKS_DEM = None              

# 温度递减率 (Gamma) 合理阈值范围 (单位: K/km)
GAMMA_MIN_K_PER_KM = 4.0
GAMMA_MAX_K_PER_KM = 9.0

# =============================================================================
# 核心处理模块
# =============================================================================

def _rename_era5_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_dict = {}
    if "valid_time" in ds.coords:
        rename_dict["valid_time"] = "time"
    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"
    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"
    return ds.rename(rename_dict) if rename_dict else ds

def load_dem_data():
    dem_ds = xr.open_dataset(DEM_PATH, engine="netcdf4")
    Hhigh = dem_ds["z_high"].clip(min=0).compute()
    target_lon = Hhigh.lon.values
    target_lat = Hhigh.lat.values
    return target_lon, target_lat, Hhigh

def build_file_list():
    return sorted([f for p in FILE_PATTERNS for f in glob.glob(os.path.join(ERA5_DIR, p)) if os.path.isfile(f)])

def make_Hlow_from_Hhigh(Hhigh_1km: xr.DataArray, era5_sample: xr.Dataset):
    Hlow = Hhigh_1km.interp(lon=era5_sample.lon, lat=era5_sample.lat, method="linear")
    return Hlow.clip(min=0).compute() # 立即计算结果，释放 Dask 内存

def load_all_era5_hourly(file_list, target_lon, target_lat):
    buf = 0.3 
    min_lon, max_lon = target_lon.min() - buf, target_lon.max() + buf
    min_lat, max_lat = target_lat.min() - buf, target_lat.max() + buf

    datasets = []
    drop_vars = ['d2m', 'skt', 'sp', 'tp', 'u10', 'v10']
    
    for idx, fp in enumerate(file_list):
        try:
            with xr.open_dataset(fp, engine="netcdf4", drop_variables=drop_vars) as ds:
                ds = _rename_era5_coords(ds)
                lat_slice = slice(max_lat, min_lat) if ds.lat[0] > ds.lat[-1] else slice(min_lat, max_lat)
                ds = ds.sel(lon=slice(min_lon, max_lon), lat=lat_slice)
                datasets.append(ds)
                if (idx + 1) % 50 == 0:
                    print(f"[数据加载] 进度: {idx+1}/{len(file_list)}")
        except Exception as e:
            print(f"[异常警告] 文件读取失败 {os.path.basename(fp)}: {e}")
            continue

    if not datasets:
        raise ValueError("无有效可读取的ERA5数据文件")

    ds = xr.concat(datasets, dim="time", join="override")
    ds = ds.chunk(CHUNKS_ERA5)
    
    t2m = ds["t2m"] - 273.15
    t2m_hourly = t2m.sel(time=slice("2025-01-01", "2025-12-31"))
    return t2m_hourly

def estimate_monthly_gamma(t2m_hourly: xr.DataArray, Hlow: xr.DataArray):
    t_month = t2m_hourly.groupby("time.month").mean("time").compute()
    mask = (Hlow > 0) & np.isfinite(Hlow)
    H = Hlow.where(mask)

    gammas = []
    for m in t_month["month"].values:
        Tm = t_month.sel(month=m).where(mask)
        H1 = H.stack(points=("lat", "lon")).compute()
        T1 = Tm.stack(points=("lat", "lon")).compute()

        valid = np.isfinite(H1) & np.isfinite(T1)
        H1v = H1.where(valid, drop=True)
        T1v = T1.where(valid, drop=True)

        if H1v.size < 50:
            gamma = 6.5 / 1000.0
        else:
            Hc = H1v - H1v.mean()
            Tc = T1v - T1v.mean()
            varH = (Hc * Hc).mean()
            covTH = (Tc * Hc).mean()
            slope = covTH / varH
            gamma = -float(slope)

        gamma_k_per_km = gamma * 1000.0
        gamma_k_per_km = float(np.clip(gamma_k_per_km, GAMMA_MIN_K_PER_KM, GAMMA_MAX_K_PER_KM))
        gamma = gamma_k_per_km / 1000.0
        gammas.append(gamma)

    gamma_month = xr.DataArray(
        np.array(gammas, dtype="float32"),
        dims=("month",),
        coords={"month": t_month["month"].values},
        name="gamma"
    )
    gamma_month.attrs["units"] = "degC_per_m"
    gamma_month.attrs["meaning"] = "positive lapse rate: temperature decreases with height"
    gamma_month.attrs["range_k_per_km"] = f"{GAMMA_MIN_K_PER_KM}~{GAMMA_MAX_K_PER_KM}"
    return gamma_month.compute()

def downscale_t2m_three_step(t2m_hourly: xr.DataArray, Hlow: xr.DataArray, Hhigh: xr.DataArray,
                             target_lon, target_lat, gamma_month: xr.DataArray, interp_method="linear"):
    gamma_for_time = gamma_month.sel(month=t2m_hourly["time"].dt.month)
    gamma_for_time_3d = gamma_for_time.broadcast_like(t2m_hourly)

    # 1. 消除地形影响 (归一化至海平面)
    Tsea = t2m_hourly + gamma_for_time_3d * Hlow
    
    # 2. 空间插值
    Tsea_1km = Tsea.interp(
        lon=target_lon, lat=target_lat, 
        method=interp_method,
        kwargs={"fill_value": "extrapolate"}
    )
    
    # 3. 引入高精度地形复原
    gamma_1km = gamma_for_time.broadcast_like(Tsea_1km)
    Tfinal = Tsea_1km - gamma_1km * Hhigh

    Tfinal.name = "t2m"
    Tfinal.attrs = {
        "units": "degree_Celsius",
        "long_name": "2m air temperature downscaled (sea-level normalization + monthly lapse rate)",
        "method": f"3-step lapse-rate downscaling: Tsea=T+Γ*Hlow, interp({interp_method}), T=Tsea-Γ*Hhigh",
        "dem_source": "GEBCO 2025 1km",
        "gamma_estimation": "monthly spatial regression on ERA5 grid with clipping",
        "gamma_range_k_per_km": f"{GAMMA_MIN_K_PER_KM}~{GAMMA_MAX_K_PER_KM}"
    }
    return Tfinal

# =============================================================================
# 主控入口
# =============================================================================
def main():
    cluster = LocalCluster(
        n_workers=4,           
        threads_per_worker=1,    
        memory_limit='60GB'      
    )
    client = Client(cluster)
    print(f"🚀 计算集群已就绪！监控地址: {client.dashboard_link}")
    print("[1/7] 加载高分辨率 DEM 数据...")
    target_lon, target_lat, Hhigh = load_dem_data()
    print(f"      目标网格维度: {len(target_lon)} x {len(target_lat)}")

    print("[2/7] 扫描 ERA5 数据集...")
    file_list = build_file_list()
    print(f"      检测到文件数量: {len(file_list)}")

    print("[3/7] 加载 ERA5 2m温度数据 (逐小时)...")
    t2m_hourly = load_all_era5_hourly(file_list, target_lon, target_lat)

    print("[4/7] 提取 ERA5 原始网格高程 (Hlow)...")
    Hlow = make_Hlow_from_Hhigh(Hhigh, t2m_hourly.isel(time=0))

    print("[5/7] 估算月度温度递减率 (Gamma)...")
    gamma_month = estimate_monthly_gamma(t2m_hourly, Hlow)

    print("[6/7] 执行三步法物理降尺度 (启动 Dask 延迟计算)...")
    t2m_1km = downscale_t2m_three_step(
        t2m_hourly=t2m_hourly,
        Hlow=Hlow,
        Hhigh=Hhigh,
        target_lon=target_lon,
        target_lat=target_lat,
        gamma_month=gamma_month,
        interp_method="linear"
    )

    print("[7/7] 导出 NetCDF 结果文件...")
    encoding = {
    "t2m": {
        "zlib": True, 
        "complevel": 1,        
        "dtype": "float32", 
        "chunksizes": (24, 851, 851), 
        "_FillValue": -9999.0
    }
}
    with ProgressBar():
        t2m_1km.to_netcdf(OUTPUT_PATH, encoding=encoding, engine="netcdf4")

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print("\n[处理完成]")
    print(f"输出路径: {OUTPUT_PATH}")
    print(f"文件大小: {size_mb:.2f} MB")
    print(f"数据形状: {t2m_1km.shape} (time, lat, lon)")
    print("✅ 全年 8760 小时超高清网格渲染完毕！")

if __name__ == "__main__":
    main()