import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import LinearSegmentedColormap
import warnings
import gc

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 基础配置
# =============================================================================
TARGET_DATE = '2025-01-15'
BASE_DIR = "/home/wangzonghan/bisheshuju"

MODEL_DIR = f"{BASE_DIR}/Results/Models_随机森林"
INFERENCE_DIR = f"/data/Machine_Learning/wangzonghan/YRD_PM25_Project/Inference_Features"
FIG_DIR = f"{BASE_DIR}/Results/Figures_随机森林"
DATASET_PATH = f"{BASE_DIR}/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"

CITY_SHP_PATH = "/home/yanchengzhu/Map/GIS/中国_市.shp"
PROVINCE_SHP_PATH = "/home/yanchengzhu/Map/GIS/中国_省.shp"

os.makedirs(FIG_DIR, exist_ok=True)

FONT_PATH = f"{BASE_DIR}/fonts/SimHei.ttf"
my_font = FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else FontProperties()

def main():
    print("="*70)
    print(f"🚀 启动 1km 时空降维反演与验证管线 | {TARGET_DATE}")
    print("="*70)

    # 1. 加载模型与特征清单
    best_rf_model = joblib.load(os.path.join(MODEL_DIR, "best_rf_model_terrain.pkl"))
    best_features = joblib.load(os.path.join(MODEL_DIR, "best_features_list_terrain.pkl"))

    # 2. 读取 1700万+ 行的 24小时超级特征矩阵
    grid_file = os.path.join(INFERENCE_DIR, f"YRD_Grid_Hourly_Features_Terrain_{TARGET_DATE.replace('-', '')}.parquet")
    if not os.path.exists(grid_file):
        print(f" ❌ 找不到对应日期的特征文件: {grid_file}")
        print(" 💡 请确认全年的特征提取脚本是否已经生成了该文件。")
        return

    print(" -> ⏳ 正在加载 24 小时超级特征矩阵 (约1700万行)...")
    grid_df = pd.read_parquet(grid_file)
    X_grid = grid_df[best_features].astype('float32')
    
    # 执行全场小时级预测
    print(f" -> 🚀 正在执行全场像素级反演...")
    grid_df['pred_hourly'] = np.clip(best_rf_model.predict(X_grid), 0, 300)

    print(" -> 🔄 正在将 24 小时预测结果折叠压缩为【日均空间分布】...")
    daily_grid_df = grid_df.groupby(['lon', 'lat'])['pred_hourly'].mean().reset_index()
    daily_grid_df.rename(columns={'pred_hourly': 'pred_daily'}, inplace=True)
    print(f"    - 折叠完成！输出单层日均网格数: {len(daily_grid_df):,}")

    # =============================================================================
    # 3. 提取 15% 独立测试站点 
    # =============================================================================
    print("\n -> 🎯 正在按照 15% 比例客观分离测试站点...")
    original_df = pd.read_parquet(DATASET_PATH)
    original_df['date'] = pd.to_datetime(original_df['date'])
    
    sites_info = original_df[['site_code', 'lon', 'lat']].drop_duplicates().reset_index(drop=True)
    num_test = int(len(sites_info) * 0.15)
    
    # 采用完全相同的空间分层聚类逻辑
    kmeans = KMeans(n_clusters=num_test, random_state=42, n_init=10)
    sites_info['spatial_cluster'] = kmeans.fit_predict(sites_info[['lon', 'lat']])
    
    test_sites_df = sites_info.groupby('spatial_cluster').apply(
        lambda x: x.sample(n=1, random_state=42)
    ).reset_index(drop=True)
    test_sites_list = test_sites_df['site_code'].tolist()
    
    # 保存完整的独立测试站底图，用于排查缺测
    test_sites_coords = test_sites_df[['site_code', 'lon', 'lat']].copy() 

    # 提取测试站点的【小时级】真实值，然后聚合为【日均真实值】
    target_dt = pd.to_datetime(TARGET_DATE).date() 
    test_day_hourly = original_df[(original_df['date'].dt.date == target_dt) & 
                                  (original_df['site_code'].isin(test_sites_list))].copy()
    
    test_day_df = test_day_hourly.groupby(['site_code', 'lon', 'lat'])['pm25_hourly'].mean().reset_index()
    test_day_df.rename(columns={'pm25_hourly': 'pm25_daily'}, inplace=True)

    # =============================================================================
    # 4. 空间查询与偏差比对
    # =============================================================================
    avg_err = np.nan
    rmse_val = np.nan
    
    if not test_day_df.empty:
        tree = cKDTree(daily_grid_df[['lon', 'lat']].values)
        _, indices = tree.query(test_day_df[['lon', 'lat']].values)
        test_day_df['extracted_pred'] = daily_grid_df['pred_daily'].iloc[indices].values
        test_day_df['error'] = test_day_df['extracted_pred'] - test_day_df['pm25_daily']
        test_day_df = test_day_df.dropna(subset=['error'])
        
        avg_err = test_day_df['error'].mean()
        rmse_val = np.sqrt(np.mean(test_day_df['error']**2))
        print(f"    - 当日独立验证站点数: {len(test_day_df)} | 均值偏差 (ME): {avg_err:.2f} | RMSE: {rmse_val:.2f}")
    else:
        print("    - ⚠️ 警告：当日测试集站点数据为空！")

    # 找出缺测站点
    valid_site_codes = test_day_df['site_code'].tolist() if not test_day_df.empty else []
    missing_sites_df = test_sites_coords[~test_sites_coords['site_code'].isin(valid_site_codes)]

    # =============================================================================
    # 5. 绘图渲染 
    # =============================================================================
    print("\n -> 🎨 正在渲染日均空间分布验证图...")
    fig = plt.figure(figsize=(12, 11), dpi=300) 
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    min_lon, max_lon = 114.5, 123.0
    min_lat, max_lat = 27.0, 35.5
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    # 绘制背景地图
    scatter_bg = ax.scatter(daily_grid_df['lon'], daily_grid_df['lat'], c=daily_grid_df['pred_daily'], 
                            cmap='Spectral_r', vmin=20, vmax=150, s=1.5, marker='s', zorder=1)
    
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black', zorder=5)

    # 渲染市级、省级高精度边界
    if os.path.exists(CITY_SHP_PATH):
        try:
            city_geom = cfeature.ShapelyFeature(shpreader.Reader(CITY_SHP_PATH).geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='dimgray', linewidth=0.5, linestyle='--', alpha=0.7)
            ax.add_feature(city_geom, zorder=3)
        except: pass
    if os.path.exists(PROVINCE_SHP_PATH):
        try:
            prov_geom = cfeature.ShapelyFeature(shpreader.Reader(PROVINCE_SHP_PATH).geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='#222222', linewidth=1.2, linestyle='-', alpha=1.0)
            ax.add_feature(prov_geom, zorder=4)
        except: pass
    
    # 海洋纯白遮罩
    ocean_mask = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='white')
    ax.add_feature(ocean_mask, edgecolor='none', zorder=5)
                               
    # 绘制有效散点误差
    if not test_day_df.empty:
        v_err = 30 
        colors_with_plateau = [
            (0.0, '#0000FF'), 
            (0.4, '#FFFFFF'),  
            (0.5, '#FFFFFF'),
            (0.6, '#FFFFFF'), 
            (1.0, '#FF0000')   
        ]
        beautiful_err_cmap = LinearSegmentedColormap.from_list("BeautifulErrorCmap", colors_with_plateau)
        scatter_err = ax.scatter(test_day_df['lon'].values, test_day_df['lat'].values, 
                                 c=test_day_df['error'].values, cmap=beautiful_err_cmap, 
                                 vmin=-v_err, vmax=v_err, s=150, alpha=1.0, 
                                 edgecolors='black', linewidths=1.2, zorder=6)
            
    # 绘制缺失站点 (打灰 X)
    if not missing_sites_df.empty:
        ax.scatter(missing_sites_df['lon'].values, missing_sites_df['lat'].values, 
                   c='lightgray', marker='X', s=100, edgecolors='dimgray', linewidths=1.0, 
                   transform=ccrs.PlateCarree(), zorder=6)

    # 右下角量化统计信息文本框
    stats_text = f"当日有效站点: {len(test_day_df)}\nME 偏差均值: {avg_err:+.2f}\nRMSE: {rmse_val:.2f}" if not test_day_df.empty else "真值暂缺"
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontproperties=my_font, fontsize=16, weight='bold',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='none'), zorder=7)

    # 调整边距 (已清理重复代码)
    plt.subplots_adjust(left=0.05, right=0.86, bottom=0.15, top=0.92)

    # 1. 下方主色标 (水平色标) - 【明确标示为底图】
    cbar_ax = fig.add_axes([0.15, 0.05, 0.55, 0.02])
    cbar1 = plt.colorbar(scatter_bg, cax=cbar_ax, orientation='horizontal')
    cbar1.set_label(r'【背景底图】长三角 1km 预测日均 PM$_{2.5}$ 浓度 ($\mu g/m^3$)', fontproperties=my_font, fontsize=18, weight='bold')
    cbar1.ax.tick_params(labelsize=14)
    
    # 2. 右侧偏差色标 (垂直色标) - 【明确标示为圆圈】
    if not test_day_df.empty:
        cbar_ax2 = fig.add_axes([0.88, 0.22, 0.02, 0.58])
        cbar2 = plt.colorbar(scatter_err, cax=cbar_ax2, orientation='vertical')
        cbar2.set_label(r'【圆圈站点】独立测试站点偏差：预测值 - 观测值 ($\mu g/m^3$)', fontproperties=my_font, fontsize=18, weight='bold')
        cbar2.ax.tick_params(labelsize=14)
    
    # 3. 坐标轴经纬度刻度
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--')
    gl.top_labels = True 
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator([115.5, 117.0, 118.5, 120.0, 121.5]) 
    gl.ylocator = mticker.FixedLocator([28.5, 30.0, 31.5, 33.0, 34.5])
    gl.xlabel_style = {'size': 14, 'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'size': 14, 'color': 'black', 'weight': 'bold'}

    # 4. 全局统一图例框 
    valid_marker = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='white', markeredgecolor='black', markersize=10, label='有效验证站点')
    missing_marker = mlines.Line2D([], [], color='white', marker='X', markerfacecolor='lightgray', markeredgecolor='dimgray', markersize=10, label='数据缺失站点')
    handles_list = [valid_marker, missing_marker]
    if os.path.exists(PROVINCE_SHP_PATH):
        handles_list.append(mlines.Line2D([], [], color='#222222', linestyle='-', linewidth=2.0, label='省级边界'))
    if os.path.exists(CITY_SHP_PATH):
        handles_list.append(mlines.Line2D([], [], color='dimgray', linestyle='--', linewidth=1.5, label='地级市边界'))
        
    legend_font = FontProperties(fname=FONT_PATH, size=15, weight='bold') if os.path.exists(FONT_PATH) else FontProperties(size=15, weight='bold')
    
    fig.legend(handles=handles_list, loc='lower right', bbox_to_anchor=(0.98, 0.02), 
               prop=legend_font, frameon=True, framealpha=1.0, 
               edgecolor='black', facecolor='white')

    # 标题
    ax.set_title(f'长三角 1km 机器学习反演日均验证图 ({TARGET_DATE})', fontproperties=my_font, fontsize=26, weight='bold', pad=25)
    
    save_path = os.path.join(FIG_DIR, f"Spatial_Daily_{TARGET_DATE.replace('-', '')}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f" -> ✅ 渲染完成！结果保存至: {save_path}")

if __name__ == "__main__":
    main()
