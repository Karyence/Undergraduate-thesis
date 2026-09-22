import xarray as xr
import matplotlib.pyplot as plt
import os
import warnings
from matplotlib.font_manager import FontProperties
import geopandas as gpd

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 1. 核心配置
# =============================================================================
NC_FILE = os.path.expanduser("~/bisheshuju/LandCover/YRD_LandCover_Fractions_1km_Final.nc")
OUTPUT_FIG = os.path.expanduser("~/bisheshuju/LandCover/FigS3_YRD_LandCover_Visualization.png")
SHP_FILE = "/home/yanchengzhu/Map/GIS/长三角.shp"

PLOT_CONFIG = [
    ('cropland_frac', 'YlOrBr', '(a) Cropland'),      
    ('forest_frac', 'Greens', '(b) Forest'),          
    ('building_frac', 'Reds', '(c) Building'),        
    ('water_frac', 'Blues', '(d) Water'),             
    ('traffic_frac', 'Oranges', '(e) Traffic'),   
    ('barren_frac', 'Greys', '(f) Barren'),           
    ('grassland_frac', 'YlGn', '(g) Grassland'),      
    ('shrubland_frac', 'BuGn', '(h) Shrubland'),    
    ('wetland_frac', 'PuBuGn', '(i) Wetland')         
]

# =============================================================================
# 2. 空间边界获取模块
# =============================================================================
def get_yrd_boundary():
    print("🗺️ 正在读取本地长三角标准行政边界 Shapefile 数据...")
    try:
        yrd_map = gpd.read_file(SHP_FILE)
        print("   ✅ 成功获取长三角标准行政边界！")
        return yrd_map
    except Exception as e:
        print(f"   ⚠️ 读取本地地图失败: {e}")
        print("   👉 将启用备用方案：通过数据掩码自动绘制外轮廓！")
        return None

# =============================================================================
# 3. 可视化主程序
# =============================================================================
def plot_landcover_fractions():
    print(f"🚀 开始加载数据: {os.path.basename(NC_FILE)}")
    ds = xr.open_dataset(NC_FILE)
    
    yrd_map = get_yrd_boundary()
    
    print("🎨 正在生成叠加边界线的高清空间分布拼图...")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 18), dpi=300)
    axes = axes.flatten()

    for i, (var_name, cmap, title) in enumerate(PLOT_CONFIG):
        ax = axes[i]
        
        dataArray = ds[var_name]
        im = dataArray.plot(
            ax=ax, 
            cmap=cmap, 
            vmin=0, vmax=1, 
            add_colorbar=False 
        )
        
        # 叠加空间边界线
        if yrd_map is not None:
            yrd_map.boundary.plot(ax=ax, edgecolor='black', linewidth=0.6, alpha=0.7)
        else:
            valid_mask = dataArray.notnull().astype(int)
            ax.contour(ds.lon, ds.lat, valid_mask, levels=[0.5], colors='black', linewidths=0.6, alpha=0.7)
        
        ax.set_title(title, fontsize=18, weight='bold', pad=12)
        ax.set_xlabel('Longitude (°E)', fontsize=15, weight='bold')
        ax.set_ylabel('Latitude (°N)', fontsize=15, weight='bold')
        ax.set_aspect('equal')
        
        ax.tick_params(axis='both', which='major', labelsize=13)
        
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.85, pad=0.04)
        cbar.set_label('Area Fraction', fontsize=15, weight='bold')
        cbar.ax.tick_params(labelsize=13) 

    plt.tight_layout(h_pad=3.0, w_pad=2.0) 

    print(f"💾 正在保存高清图像至: {OUTPUT_FIG}")
    os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)
    plt.savefig(OUTPUT_FIG, bbox_inches='tight')
    plt.close()
    
    print("✅ 完美出图！带有行政边界的版本已经生成。")

if __name__ == "__main__":
    plot_landcover_fractions()