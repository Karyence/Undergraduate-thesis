import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 🌟 全局学术字体配置：使用弹性衬线体族 (Serif Family) 完美替代，防止 Linux 服务器报错
# =============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = "/home/wangzonghan/bisheshuju"
DATA_FILE = f"{BASE_DIR}/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"
GEBCO_PATH = f"{BASE_DIR}/GEBCO/GEBCO_2025_YRD_1km_with_slope.nc"

OUTPUT_DIR = f"{BASE_DIR}/Results/Figures_正文插图"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("="*70)
    print("🚀 开始绘制 SCI级 图 2.1 研究区域与站点分布示意图 (纯英文版)")
    print("  -> 启用全局 Serif 衬线字体 + 完美防截断外置排版")
    print("="*70)

    df = pd.read_parquet(DATA_FILE)
    sites_info = df[['site_code', 'lon', 'lat']].drop_duplicates().reset_index(drop=True)
    num_sites = len(sites_info)
    print(f"  ✅ 成功提取 {num_sites} 个监测站点。")

    # 扩大画布宽度，为右侧的长英文文本框留足空间
    fig = plt.figure(figsize=(14, 10), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    min_lon, max_lon = 114.5, 123.0
    min_lat, max_lat = 27.0, 35.5
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # 截取 terrain 色带的纯陆地部分 (跳过前25%的蓝色海洋)
    terrain_cmap = plt.get_cmap('terrain')
    terrain_colors = terrain_cmap(np.linspace(0.25, 1.0, 256))
    custom_land_cmap = mcolors.LinearSegmentedColormap.from_list('terrain_land_only', terrain_colors)

    if os.path.exists(GEBCO_PATH):
        ds = xr.open_dataset(GEBCO_PATH)
        elev_yrd = ds['DEM'].sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
        if len(elev_yrd.lat) == 0:
            elev_yrd = ds['DEM'].sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))

        lon_grid, lat_grid = np.meshgrid(elev_yrd.lon, elev_yrd.lat)
        
        # 渲染底图，vmin=0, vmax=1500 (完美切除负值导致的错误颜色)
        mesh = ax.pcolormesh(lon_grid, lat_grid, elev_yrd.values, 
                             cmap=custom_land_cmap, vmin=0, vmax=1500, 
                             transform=ccrs.PlateCarree(), shading='auto', zorder=1)
        
        # 底部颜色带条，继承全局衬线字体
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.06, aspect=35, shrink=0.9)
        cbar.set_label('Land Elevation (m)', fontsize=14, weight='bold')
        cbar.ax.tick_params(labelsize=12)

    # 【海洋遮罩】把海拔为 0 的沿海/海洋平滑遮住，纯白洗底
    ocean_mask = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='white')
    ax.add_feature(ocean_mask, edgecolor='none', zorder=2)

    # 添加海岸线和省界
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black', zorder=3)
    try:
        provinces = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
        ax.add_feature(provinces, edgecolor='dimgray', linewidth=0.8, linestyle='--', zorder=4)
    except Exception:
        pass

    # 绘制网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.6, color='gray', alpha=0.4, linestyle='--', zorder=4) 
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator([115, 116, 117, 118, 119, 120, 121, 122])
    gl.ylocator = mticker.FixedLocator([28.5, 30.0, 31.5, 33.0, 34.5])
    gl.xlabel_style = {'size': 13, 'color': 'black', 'family': 'serif'}
    gl.ylabel_style = {'size': 13, 'color': 'black', 'family': 'serif'}
    
    # 研究区域红框
    rect = Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat, 
                     linewidth=2.5, edgecolor='red', facecolor='none', 
                     transform=ccrs.PlateCarree(), zorder=5)
    ax.add_patch(rect)

    # 绘制国控监测站散点
    ax.scatter(sites_info['lon'], sites_info['lat'], c='#E74C3C', s=60, alpha=0.95, 
               edgecolors='black', linewidths=0.8, transform=ccrs.PlateCarree(), zorder=6)

    # 物理空间预留
    plt.subplots_adjust(right=0.72) 

    # 外置统计文本框
    text_str = f"Study Area: Yangtze River Delta (YRD)\nSpatial Resolution: 1 km\nMonitoring Sites: {num_sites}"
    props = dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='black', alpha=0.9)
    
    ax.text(1.025, 0.22, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='left', bbox=props, zorder=10)

    # 外置图例框
    red_box_line = mlines.Line2D([], [], color='red', linewidth=2.5, label='Study Area Boundary')
    site_marker = mlines.Line2D([], [], color='none', marker='o', markerfacecolor='#E74C3C',
                                 markeredgecolor='black', markersize=9, label='Monitoring Station')

    leg = ax.legend(handles=[red_box_line, site_marker], loc='lower left',
                    bbox_to_anchor=(1.025, 0.03), prop={'size': 12}, 
                    framealpha=0.9, edgecolor='black', borderaxespad=0., labelspacing=1.0)
    leg.set_zorder(10)

    save_path = os.path.join(OUTPUT_DIR, "Fig_2.1_Study_Area_and_Sites_Terrain_EN.png")
    
    # 增加 pad_inches=0.2 参数，强制保留物理白边保护层，防止卡边
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.close()
    
    print(f"\n🎉 完美！纯英文衬线体版插图已成功保存至:\n👉 {save_path}")

if __name__ == "__main__":
    main()