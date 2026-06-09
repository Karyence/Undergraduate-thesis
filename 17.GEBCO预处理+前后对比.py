import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas as gpd
import warnings 

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 核心配置区域
# =============================================================================
CONF = {
    "GEBCO": "/data/Environment/Landscape/GEBCO_2025.nc",
    "SHP": "/home/yanchengzhu/Map/GIS/YRD.shp",
    "OUT_NC": os.path.expanduser("~/bisheshuju/GEBCO/GEBCO_2025_YRD_1km.nc"),
    "OUT_IMG": os.path.expanduser("~/bisheshuju/GEBCO/GEBCO_2025_YRD_1km_Map.png"),
    "RANGE": [114.5, 123.0, 27.0, 35.5], 
    "RES": 0.01,
    "COLORBAR_RANGE": [-500, 1500]  
}

# 字体加载与路径兼容
FONT_PATH = os.path.expanduser("~/bisheshuju/fonts/SimHei.ttf")
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/home/wangzonghan/bisheshuju/fonts/SimHei.ttf"

# =============================================================================
# 2. 统计对比工具函数
# =============================================================================
def print_comparison_stats(ds_old, ds_new):
    """打印地形重采样前后的统计对比分析报告"""
    print("\n" + "="*50)
    print("📊 数据处理前后对比报告")
    print("="*50)
    
    var_old = 'elevation' if 'elevation' in ds_old else list(ds_old.data_vars)[0]
    var_new = 'z_high' if 'z_high' in ds_new else list(ds_new.data_vars)[0]

    res_old_lat = abs(ds_old.lat[1] - ds_old.lat[0]).values
    res_new_lat = abs(ds_new.lat[1] - ds_new.lat[0]).values

    stats_old = {
        "min": float(ds_old[var_old].min()),
        "max": float(ds_old[var_old].max()),
        "mean": float(ds_old[var_old].mean())
    }
    stats_new = {
        "min": float(ds_new[var_new].min()),
        "max": float(ds_new[var_new].max()),
        "mean": float(ds_new[var_new].mean())
    }

    size_old = f"{ds_old.sizes['lon']}x{ds_old.sizes['lat']}"
    size_new = f"{ds_new.sizes['lon']}x{ds_new.sizes['lat']}"

    print(f"{'指标':<15} | {'原始数据 (Raw)':<20} | {'重采样后 (1km)':<20}")
    print("-" * 65)
    print(f"{'网格尺寸':<15} | {size_old:<20} | {size_new:<20}")
    print(f"{'分辨率 (Lat)':<15} | {res_old_lat:.5f}°{'':<13} | {res_new_lat:.5f}°{'':<13}")
    print(f"{'最小值 (m)':<15} | {stats_old['min']:<20.2f} | {stats_new['min']:<20.2f}")
    print(f"{'最大值 (m)':<15} | {stats_old['max']:<20.2f} | {stats_new['max']:<20.2f}")
    print(f"{'平均值 (m)':<15} | {stats_old['mean']:<20.2f} | {stats_new['mean']:<20.2f}")
    print("="*50 + "\n")

# =============================================================================
# 3. 空间可视化函数
# =============================================================================
def plot_map(ds, gdf, lons, lats, font):
    """生成带有行政边界与自定义色带的高清地形图"""
    num_x, num_y = ds.sizes['lon'], ds.sizes['lat']
    min_lon, max_lon, min_lat, max_lat = CONF["RANGE"]
    vmin, vmax = CONF["COLORBAR_RANGE"] 
    
    title_font = font.copy(); title_font.set_size(15); title_font.set_weight('bold')
    text_font = font.copy(); text_font.set_size(12)

    # 自定义地形色带 (海蓝 -> 平原绿 -> 高山棕)
    color_nodes = [
        (0.00, '#0a2342'),  # -500m: 深海深蓝
        (0.24, '#73a5c6'),  # -20m:  近海浅蓝
        (0.25, '#a3c9a8'),  # 0m:    海岸线/平原起始 (青绿)
        (0.35, '#d4e09b'),  # 200m:  丘陵过渡 (黄绿)
        (0.60, '#e5b181'),  # 700m:  中海拔山地 (暖棕)
        (0.85, '#b5651d'),  # 1200m: 高海拔山地 (深棕)
        (1.00, '#f2e8dc')   # 1500m: 顶峰 (灰白)
    ]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_terrain', color_nodes)

    fig = plt.figure(figsize=(14, 10)) 
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    im = ds['z_high'].plot(
        ax=ax, transform=ccrs.PlateCarree(), cmap=custom_cmap,  
        add_colorbar=False, vmin=vmin, vmax=vmax        
    )
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05, orientation='horizontal', location='bottom') 
    cbar.set_label('陆地海拔高度 (m) / 海洋深度 (<0m)', fontproperties=text_font, fontsize=12)
    cbar.set_ticks(np.arange(vmin, vmax+1, 250))

    ax.add_geometries(gdf.geometry, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=1.5)
    
    for lon in lons[::10]: ax.plot([lon, lon], [min_lat, max_lat], color='lightgray', lw=0.5, alpha=0.6, transform=ccrs.PlateCarree())
    for lat in lats[::10]: ax.plot([min_lon, max_lon], [lat, lat], color='lightgray', lw=0.5, alpha=0.6, transform=ccrs.PlateCarree())
    
    gl = ax.gridlines(draw_labels=True, linestyle='-', linewidth=1, color='gray', alpha=0.5)
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = gl.ylabel_style = {'fontsize': 11, 'weight': 'bold'}
    
    ax.plot([min_lon, max_lon, max_lon, min_lon, min_lon], 
            [min_lat, min_lat, max_lat, max_lat, min_lat], 
            color='red', lw=2.5, alpha=0.9, transform=ccrs.PlateCarree(), label='1km 网格边界')
    
    ax.legend(
        loc='lower left', bbox_to_anchor=(0, 1.01), 
        prop=text_font, frameon=True, framealpha=1, borderaxespad=0
    )

    grid_text = f"1km 网格统计:\nX(经度) = {num_x} 个\nY(纬度) = {num_y} 个"
    ax.text(
        1.03, 0.0, grid_text, transform=ax.transAxes, fontproperties=text_font, 
        ha='left', va='bottom', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
    )
    
    ax.set_title("长三角 (YRD) 1km 地形分布图", fontproperties=title_font, pad=35) 
    ax.set_extent([min_lon-0.1, max_lon+0.1, min_lat-0.1, max_lat+0.1])
    
    os.makedirs(os.path.dirname(CONF["OUT_IMG"]), exist_ok=True)
    plt.savefig(CONF["OUT_IMG"], dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 地图已保存: {CONF['OUT_IMG']}")
    plt.close()

# =============================================================================
# 4. 主控程序
# =============================================================================
def main():
    if os.path.exists(FONT_PATH):
        font_prop = fm.FontProperties(fname=FONT_PATH, size=12)
    else:
        print(f"⚠️ 找不到中文字体，将使用系统默认字体。")
        font_prop = fm.FontProperties(size=12)
        
    plt.rcParams['axes.unicode_minus'] = False 

    print("🚀 开始处理地形数据...")
    
    ds_raw_full = xr.open_dataset(CONF["GEBCO"], chunks={"lat": 4000, "lon": 4000})
    min_lon, max_lon, min_lat, max_lat = CONF["RANGE"]
    lat_slice = slice(max_lat, min_lat) if ds_raw_full.lat[0] > ds_raw_full.lat[-1] else slice(min_lat, max_lat)
    
    ds_raw = ds_raw_full.sel(lon=slice(min_lon, max_lon), lat=lat_slice)
    
    print("⚙️ 执行 3x3 空间平滑与 1km 重采样...")
    ds_smooth = ds_raw.rolling(lat=3, lon=3, center=True, min_periods=1).mean()
    
    new_lon = np.arange(min_lon, max_lon + CONF["RES"]/2, CONF["RES"])
    new_lat = np.arange(min_lat, max_lat + CONF["RES"]/2, CONF["RES"])
    ds_1km = ds_smooth.interp(lon=new_lon, lat=new_lat, method='linear')
    
    if 'elevation' in ds_1km: 
        ds_1km = ds_1km.rename({'elevation': 'z_high'})

    print_comparison_stats(ds_raw, ds_1km)

    os.makedirs(os.path.dirname(CONF["OUT_NC"]), exist_ok=True)
    enc = {'z_high': {'zlib': True, 'complevel': 5, 'dtype': 'float32', '_FillValue': -9999.0}}
    ds_1km.to_netcdf(CONF["OUT_NC"], encoding=enc)
    print(f"✅ 降尺度地形数据已保存: {CONF['OUT_NC']}")

    print("🎨 正在绘制地形高程分布图...")
    gdf = gpd.read_file(CONF["SHP"]).to_crs("EPSG:4326")
    plot_map(ds_1km, gdf, new_lon, new_lat, font_prop)

if __name__ == "__main__":
    main()