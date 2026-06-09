import xarray as xr
import matplotlib.pyplot as plt
import os
import warnings
from matplotlib.font_manager import FontProperties
import geopandas as gpd

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 核心配置
# =============================================================================
NC_FILE = os.path.expanduser("~/bisheshuju/LandCover/YRD_LandCover_Fractions_1km_Final.nc")
OUTPUT_FIG = os.path.expanduser("~/bisheshuju/LandCover/YRD_LandCover_Visualization_with_Boundaries.png")
SHP_FILE = "/home/yanchengzhu/Map/GIS/长三角.shp"

# 字体加载与多用户路径兼容
FONT_PATH = os.path.expanduser("~/bisheshuju/fonts/SimHei.ttf")
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/home/wangzonghan/bisheshuju/fonts/SimHei.ttf"

if os.path.exists(FONT_PATH):
    my_font = FontProperties(fname=FONT_PATH)
else:
    print(f"⚠️ 找不到字体文件 {FONT_PATH}，中文可能仍会显示为方块！")
    my_font = FontProperties() 

plt.rcParams['axes.unicode_minus'] = False

# 子图配置映射 (变量名, 颜色映射, 标题)
PLOT_CONFIG = [
    ('cropland_frac', 'YlOrBr', '耕地占比 (Cropland)'),      
    ('forest_frac', 'Greens', '林地占比 (Forest)'),          
    ('building_frac', 'Reds', '建筑占比 (Building)'),        
    ('water_frac', 'Blues', '水体占比 (Water)'),             
    ('traffic_frac', 'Oranges', '交通道路占比 (Traffic)'),   
    ('barren_frac', 'Greys', '裸地占比 (Barren)'),           
    ('grassland_frac', 'YlGn', '草地占比 (Grassland)'),      
    ('shrubland_frac', 'BuGn', '灌木地占比 (Shrubland)'),    
    ('wetland_frac', 'PuBuGn', '湿地占比 (Wetland)')         
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
        
        ax.set_title(title, fontproperties=my_font, fontsize=20, pad=15)
        ax.set_xlabel('Longitude (°E)', fontsize=16)
        ax.set_ylabel('Latitude (°N)', fontsize=16)
        ax.set_aspect('equal')
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.85, pad=0.04)
        cbar.set_label('Area Fraction', fontsize=16)
        cbar.ax.tick_params(labelsize=14) 
    fig.suptitle('长三角地区 1km 分辨率土地覆盖精细化占比空间分布', 
                 fontproperties=my_font, fontsize=30, y=0.98, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0, w_pad=2.0) 

    print(f"💾 正在保存高清图像至: {OUTPUT_FIG}")
    os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)
    plt.savefig(OUTPUT_FIG, bbox_inches='tight')
    plt.close()
    
    print("✅ 完美出图！带有行政边界的版本已经生成。")

if __name__ == "__main__":
    plot_landcover_fractions()