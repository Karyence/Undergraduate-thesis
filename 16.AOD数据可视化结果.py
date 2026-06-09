import xarray as xr
import matplotlib.pyplot as plt
import os
import warnings
import geopandas as gpd
from matplotlib.font_manager import FontProperties
import rasterio.features
from rasterio.transform import from_origin

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 核心配置路径与参数
# =============================================================================
NC_FILE = os.path.expanduser("~/bisheshuju/AOD/YRD_AOD_Daily_1km_2025.nc")
OUTPUT_FIG = os.path.expanduser("~/bisheshuju/AOD/YRD_AOD_Check_Plot.png")
SHP_FILE = "/home/yanchengzhu/Map/GIS/长三角.shp"

# 字体加载与多用户路径兼容
FONT_PATH = os.path.expanduser("~/bisheshuju/fonts/SimHei.ttf")
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/home/wangzonghan/bisheshuju/fonts/SimHei.ttf"

if os.path.exists(FONT_PATH):
    my_font = FontProperties(fname=FONT_PATH)
else:
    print("⚠️ 找不到中文字体，将使用系统默认字体。")
    my_font = FontProperties()
    
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 2. 空间边界获取模块 
# =============================================================================
def get_yrd_boundary():
    print("🗺️ 正在获取长三角标准行政边界...")
    try:
        # 读取本地矢量边界数据代替网络请求
        yrd_map = gpd.read_file(SHP_FILE)
        return yrd_map
    except Exception as e:
        print(f"  ⚠️ 获取边界失败: {e}")
        return None

# =============================================================================
# 3. 可视化主程序
# =============================================================================
def plot_aod_check():
    print(f"🚀 开始加载 AOD 数据: {os.path.basename(NC_FILE)}")
    if not os.path.exists(NC_FILE):
        print("❌ 找不到 AOD 文件，请确认路径！")
        return

    ds = xr.open_dataset(NC_FILE)
    yrd_map = get_yrd_boundary()

    print("🗺️ 正在利用长三角行政边界生成精确矢量掩膜 (完美剔除外海，保留内陆河湖)...")
    lon = ds['lon'].values
    lat = ds['lat'].values
    
    res_x = (lon[-1] - lon[0]) / (len(lon) - 1)
    res_y = (lat[-1] - lat[0]) / (len(lat) - 1)
    
    # 构建仿射变换矩阵
    transform = from_origin(lon[0] - res_x/2, lat[-1] + res_y/2, res_x, res_y)
    
    # 将矢量多边形转换为栅格掩膜
    out_mask = rasterio.features.geometry_mask(
        yrd_map.geometry, 
        out_shape=(len(lat), len(lon)), 
        transform=transform, 
        invert=False 
    )
    
    # 纬度方向对齐并反转掩膜逻辑
    land_mask = ~out_mask[::-1, :]

    # 提取代表性日期切片
    date1 = "2025-04-15"
    date2 = "2025-12-20"
    
    print(f"🔍 正在提取 {date1} 和 {date2} 的切片数据，并应用矢量掩膜...")
    try:
        aod_day1 = ds['aod'].sel(time=date1).where(land_mask)
        aod_day2 = ds['aod'].sel(time=date2).where(land_mask)
    except Exception as e:
        print(f"❌ 提取日期失败: {e}，请检查时间维度。")
        return

    print("🎨 正在生成 AOD 空间分布检验图...")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7), dpi=300)
    
    cmap = 'YlOrRd'  
    vmin, vmax = 0.0, 1.2  

    # 绘制第一幅图
    ax1 = axes[0]
    im1 = aod_day1.plot(ax=ax1, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    ax1.set_title(f'AOD 空间分布 ({date1})', fontproperties=my_font, fontsize=16, pad=12)
    
    # 绘制第二幅图
    ax2 = axes[1]
    im2 = aod_day2.plot(ax=ax2, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    ax2.set_title(f'AOD 空间分布 ({date2})', fontproperties=my_font, fontsize=16, pad=12)

    # 统一设置坐标轴和行政边界
    for ax in axes:
        if yrd_map is not None:
            yrd_map.boundary.plot(ax=ax, edgecolor='black', linewidth=0.8, alpha=0.8)
        
        ax.set_xlabel('Longitude (°E)', fontsize=12)
        ax.set_ylabel('Latitude (°N)', fontsize=12)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        ax.set_ylim(27.0, 35.5)
        ax.set_xlim(114.5, 123.0)

    # 共享同一个 Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax, orientation='vertical')
    cbar.set_label('AOD (Aerosol Optical Depth @ 0.55μm)', fontsize=12)

    fig.suptitle('长三角地区 2025 年高分辨率 AOD 时空分布', 
                 fontproperties=my_font, fontsize=22, y=0.98, fontweight='bold')

    plt.subplots_adjust(wspace=0.1, right=0.9)
    
    os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)
    print(f"💾 正在保存高清图像至: {OUTPUT_FIG}")
    plt.savefig(OUTPUT_FIG, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ 完美出图！")

if __name__ == "__main__":
    plot_aod_check()