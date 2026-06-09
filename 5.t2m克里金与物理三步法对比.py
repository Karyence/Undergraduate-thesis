import xarray as xr
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pykrige.ok import OrdinaryKriging
import warnings
import matplotlib.font_manager as fm

warnings.filterwarnings("ignore")

font_path = '/home/wangzonghan/bisheshuju/fonts/SimHei.ttf'
fm.fontManager.addfont(font_path)
custom_font = fm.FontProperties(fname=font_path)
font_name = custom_font.get_name()

plt.rcParams['font.sans-serif'] = [font_name]
plt.rcParams['axes.unicode_minus'] = False 

# =============================================================================
# 核心配置
# =============================================================================
OUTPUT_DIR = os.path.expanduser("~/bisheshuju/ERA5_LAND")
ERA5_DOWN_FILE = os.path.join(OUTPUT_DIR, "ERA5_t2m_1km_downscaled.nc")
DEM_PATH = os.path.expanduser("~/bisheshuju/GEBCO/GEBCO_2025_YRD_1km.nc")

# 选取 4 个代表性时刻
SCENARIOS = {
    "深冬寒夜": 338,     # 1月15日 02:00
    "盛夏午后": 4358,    # 7月1日 14:00
    "初春正午": 2172,    # 4月1日 12:00
    "深秋傍晚": 6570     # 10月1日 18:00 
}

def extract_era5_coarse_points(t2m_1km_ds, dem_ds, hour_index):
    """提取指定小时的粗分辨率数据"""
    t2m_hour = t2m_1km_ds['t2m'].isel(time=hour_index)
    h_high = dem_ds['z_high'].clip(min=0)
    
    sample_step = 25
    lons = t2m_hour.lon.values[::sample_step]
    lats = t2m_hour.lat.values[::sample_step]
    
    t_coarse = t2m_hour.values[::sample_step, ::sample_step].flatten()
    h_coarse = h_high.values[::sample_step, ::sample_step].flatten()
    
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    lon_coarse = lon_mesh.flatten()
    lat_coarse = lat_mesh.flatten()
    
    valid = ~np.isnan(t_coarse)
    return lon_coarse[valid], lat_coarse[valid], h_coarse[valid], t_coarse[valid]

def perform_regression_kriging(lon_c, lat_c, h_c, t_c, target_lon, target_lat, h_high):
    """执行回归克里金 (Regression Kriging) 空间插值"""
    start_time = time.time()
    
    reg = LinearRegression()
    reg.fit(h_c.reshape(-1, 1), t_c)
    lapse_rate = -reg.coef_[0] * 1000
    
    t_pred_c = reg.predict(h_c.reshape(-1, 1))
    residuals_c = t_c - t_pred_c
    rmse_rk = np.sqrt(np.mean(residuals_c**2))
    
    print("   -> 正在进行克里金空间插值 ...")
    ok = OrdinaryKriging(
        lon_c, lat_c, residuals_c, 
        variogram_model='spherical',
        nlags=15
    )
    residuals_1km, _ = ok.execute('grid', target_lon, target_lat, backend='loop')
    
    h_high_flat = h_high.values.flatten()
    valid_mask = ~np.isnan(h_high_flat) 
    
    t_trend_flat = np.full_like(h_high_flat, np.nan, dtype=float) 
    t_trend_flat[valid_mask] = reg.predict(h_high_flat[valid_mask].reshape(-1, 1))
    t_trend_1km = t_trend_flat.reshape(h_high.shape) 
    
    t_rk_1km = t_trend_1km + residuals_1km.data
    
    print(f"   ✅ RK 处理完成，耗时: {time.time() - start_time:.1f} 秒")
    return t_rk_1km, lapse_rate, rmse_rk

def main():
    print("="*70)
    print("🔬 物理三步法 vs 回归克里金(RK) 极值压缩效应 全场景对比实验")
    print("="*70)
    
    print("🌍 正在加载全局数据环境 ...")
    t2m_ds = xr.open_dataset(ERA5_DOWN_FILE)
    dem_ds = xr.open_dataset(DEM_PATH)
    
    target_lon = dem_ds.lon.values
    target_lat = dem_ds.lat.values
    h_high = dem_ds['z_high'].clip(min=0)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for scenario_name, hour_index in SCENARIOS.items():
        print("\n" + "▼"*70)
        print(f"🚀 当前处理场景: {scenario_name}")
        
        t_3step_1km = t2m_ds['t2m'].isel(time=hour_index).compute().values
        time_val = t2m_ds['time'].isel(time=hour_index).values
        
        date_str = np.datetime_as_string(time_val, unit='h').replace('T', ' ') + ":00"
        print(f"   -> 具体时刻: {date_str}")
        
        lon_c, lat_c, h_c, t_c = extract_era5_coarse_points(t2m_ds, dem_ds, hour_index)
        
        t_rk_1km, lapse_rate, rmse_rk = perform_regression_kriging(
            lon_c, lat_c, h_c, t_c, target_lon, target_lat, h_high
        )
        ocean_mask = np.isnan(t_3step_1km)
        t_rk_1km[ocean_mask] = np.nan
        
        min_3step, max_3step, mean_3step = np.nanmin(t_3step_1km), np.nanmax(t_3step_1km), np.nanmean(t_3step_1km)
        min_rk, max_rk, mean_rk = np.nanmin(t_rk_1km), np.nanmax(t_rk_1km), np.nanmean(t_rk_1km)
        
        # ==========================================
        # 绘图部分
        # ==========================================
        TITLE_FS = 16   
        LABEL_FS = 14   
        TICK_FS = 12    
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6)) 
        
        suptitle_text = f"气象场景：{scenario_name}  |  时间：{date_str}"
        fig.suptitle(suptitle_text, fontsize=20, fontweight='bold', y=0.98)
        
        vmin = min(min_3step, min_rk)
        vmax = max(max_3step, max_rk)
        
        im0 = axes[0].imshow(t_3step_1km, cmap='RdYlBu_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title(f"物理三步法降尺度 (Physical 3-Step)\n(Min: {min_3step:.2f}℃)", fontsize=TITLE_FS)
        axes[0].tick_params(labelsize=TICK_FS)
        cb0 = plt.colorbar(im0, ax=axes[0])
        cb0.set_label('温度 (℃)', fontsize=LABEL_FS)
        cb0.ax.tick_params(labelsize=TICK_FS)
        
        im1 = axes[1].imshow(t_rk_1km, cmap='RdYlBu_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title(f"回归克里金插值 (Regression Kriging)\n(Min: {min_rk:.2f}℃)", fontsize=TITLE_FS)
        axes[1].tick_params(labelsize=TICK_FS)
        cb1 = plt.colorbar(im1, ax=axes[1])
        cb1.set_label('温度 (℃)', fontsize=LABEL_FS)
        cb1.ax.tick_params(labelsize=TICK_FS)
        
        diff_map = t_3step_1km - t_rk_1km
        im2 = axes[2].imshow(diff_map, cmap='seismic', origin='lower', vmin=-3, vmax=3)
        axes[2].set_title("空间差异图\n(三步法 - RK法)", fontsize=TITLE_FS)
        axes[2].tick_params(labelsize=TICK_FS)
        cb2 = plt.colorbar(im2, ax=axes[2])
        cb2.set_label('温差 (℃)', fontsize=LABEL_FS)
        cb2.ax.tick_params(labelsize=TICK_FS)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        output_img = os.path.join(OUTPUT_DIR, f"Comparison_{scenario_name}.png")
        plt.savefig(output_img, dpi=300)
        plt.close(fig) 
        print(f"📸 图表已保存: {output_img}")
        
    print("\n" + "="*70)
    print(f"🎉 所有处理完毕！请查看 {OUTPUT_DIR} 目录！")

if __name__ == "__main__":
    main()