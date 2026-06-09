import os
import glob
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
from matplotlib.colors import LinearSegmentedColormap
import warnings
import gc 

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 基础配置与路径 
# =============================================================================
BASE_DIR = "/home/wangzonghan/bisheshuju"
MODEL_DIR = f"{BASE_DIR}/Results/Models_随机森林"
INFERENCE_DIR = "/data/Machine_Learning/wangzonghan/YRD_PM25_Project/Inference_Features"
FIG_DIR = f"{BASE_DIR}/Results/Figures_随机森林"
DATASET_PATH = f"{BASE_DIR}/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"

CITY_SHP_PATH = "/home/yanchengzhu/Map/GIS/中国_市.shp"
PROVINCE_SHP_PATH = "/home/yanchengzhu/Map/GIS/中国_省.shp"

os.makedirs(FIG_DIR, exist_ok=True)

FONT_PATH = f"{BASE_DIR}/fonts/SimHei.ttf"
my_font = FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else FontProperties()

TARGET_MONTHS = range(1, 13)  # 遍历 1月 到 12月

def main():
    print("="*70)
    print("🚀 启动 2025 年 1-12 月 2D 空间连续反演与 NMB 标准化平均偏差全局组图管线")
    print("="*70)

    # =============================================================================
    # 1. 加载预训练模型与提取独立测试集
    # =============================================================================
    print(" -> [1/3] 加载预训练模型并精确复现 15% 独立测试站点隔离...")
    model_path = os.path.join(MODEL_DIR, "best_rf_model_terrain.pkl")
    features_path = os.path.join(MODEL_DIR, "best_features_list_terrain.pkl")
    
    if not os.path.exists(model_path):
        print(f" ❌ 找不到预训练模型，请检查路径: {model_path}")
        return
        
    best_rf_model = joblib.load(model_path)
    best_features = joblib.load(features_path)

    original_df = pd.read_parquet(DATASET_PATH)
    original_df['date'] = pd.to_datetime(original_df['date'])
    original_df['month'] = original_df['date'].dt.month
    
    sites_info = original_df[['site_code', 'lon', 'lat']].drop_duplicates().reset_index(drop=True)
    num_test = int(len(sites_info) * 0.15)
    kmeans = KMeans(n_clusters=num_test, random_state=42, n_init=10)
    sites_info['spatial_cluster'] = kmeans.fit_predict(sites_info[['lon', 'lat']])
    
    test_sites_df = sites_info.groupby('spatial_cluster').apply(
        lambda x: x.sample(n=1, random_state=42)
    ).reset_index(drop=True)
    
    test_sites_list = test_sites_df['site_code'].tolist()
    test_sites_coords = test_sites_df[['site_code', 'lon', 'lat']].copy() 
    
    test_df = original_df[original_df['site_code'].isin(test_sites_list)].copy()

    # =============================================================================
    # 2. 定制全局色标跨度设定
    # =============================================================================
    VMIN_BG, VMAX_BG = 10, 80   # PM2.5绝对浓度范围
    V_ERR_NMB = 50              # NMB 百分比误差范围设定为 ±50% 
    cmap_err = plt.cm.RdBu_r    # 发散色标：红高估，蓝低估

    # =============================================================================
     # =============================================================================
    print("\n -> [2/3] 创建全局 3x4 矩阵画布并开始逐月聚合推理渲染...")
    
    fig = plt.figure(figsize=(20, 26), dpi=300) 
    fig.suptitle('长三角 1km 空间反演与独立验证 NMB 标准化平均偏差演变序列 (2025年 1-12月)', 
                 fontproperties=my_font, fontsize=36, weight='bold', y=0.94)
                 
    sm_bg = None
    sm_err = None

    for idx, target_month in enumerate(TARGET_MONTHS):
        print(f"\n   ⏳ 正在计算面板 [{idx+1}/12]: 2025 年 {target_month:02d} 月...")
        
        month_str = f"2025{target_month:02d}"
        grid_files_pattern = f"{INFERENCE_DIR}/YRD_Grid_Hourly_Features_Terrain_{month_str}*.parquet"
        grid_files = glob.glob(grid_files_pattern)
        
        if not grid_files:
            print(f"      ⚠️ 跳过: 缺失 {month_str} 的网格特征文件。")
            continue
            
        monthly_pred_sum = None
        pixel_valid_counts = None  
        grid_lon_lat = None
        
        for daily_file in grid_files:
            grid_df = pd.read_parquet(daily_file)
            X_grid = grid_df[best_features].astype('float32')
            
            # 使用最快的原生预测
            grid_df['pred_hourly'] = best_rf_model.predict(X_grid)
            
            # 空间折叠为日均
            daily_grid = grid_df.groupby(['lon', 'lat'])['pred_hourly'].mean().reset_index()
            daily_grid = daily_grid.sort_values(by=['lat', 'lon']).reset_index(drop=True)
            
            if monthly_pred_sum is None: 
                grid_lon_lat = daily_grid[['lon', 'lat']].copy()
                monthly_pred_sum = np.zeros(len(daily_grid), dtype=float)
                pixel_valid_counts = np.zeros(len(daily_grid), dtype=float)
                
            monthly_pred_sum += daily_grid['pred_hourly'].values
            pixel_valid_counts += 1
            
            del grid_df, X_grid, daily_grid
            gc.collect()
            
        with np.errstate(divide='ignore', invalid='ignore'):
            grid_monthly_mean = np.where(pixel_valid_counts > 0, 
                                         monthly_pred_sum / pixel_valid_counts, 
                                         np.nan)
        grid_lon_lat['monthly_pm25'] = np.clip(grid_monthly_mean, 0, 150)

        # =========================================================================
        # 基于 cKDTree 计算独立测试站点的 NMB
        # =========================================================================
        test_month_df = test_df[test_df['month'] == target_month].copy()
        site_monthly_stats = pd.DataFrame()
        
        if not test_month_df.empty:
            site_monthly_stats = test_month_df.groupby(['site_code', 'lon', 'lat'])['pm25_hourly'].mean().reset_index()
            site_monthly_stats.rename(columns={'pm25_hourly': 'obs_mean'}, inplace=True)
            
            valid_grid = grid_lon_lat.dropna(subset=['monthly_pm25']).reset_index(drop=True)
            if not valid_grid.empty:
                tree = cKDTree(valid_grid[['lon', 'lat']].values)
                _, indices = tree.query(site_monthly_stats[['lon', 'lat']].values)
                site_monthly_stats['pred_mean'] = valid_grid['monthly_pm25'].iloc[indices].values
                
                # 计算标准化平均偏差 NMB (%) 
                site_monthly_stats['NMB'] = (site_monthly_stats['pred_mean'] - site_monthly_stats['obs_mean']) / site_monthly_stats['obs_mean'] * 100
                site_monthly_stats = site_monthly_stats.dropna(subset=['NMB'])

        valid_site_codes = site_monthly_stats['site_code'].tolist() if not site_monthly_stats.empty else []
        missing_sites_df = test_sites_coords[~test_sites_coords['site_code'].isin(valid_site_codes)]

        # =========================================================================
        # 绘图渲染模块 
        # =========================================================================
        ax = fig.add_subplot(4, 3, idx + 1, projection=ccrs.PlateCarree())
        ax.set_extent([114.5, 123.0, 27.0, 35.5], crs=ccrs.PlateCarree())
        
        scatter_bg = ax.scatter(grid_lon_lat['lon'], grid_lon_lat['lat'], c=grid_lon_lat['monthly_pm25'], 
                                cmap='Spectral_r', vmin=VMIN_BG, vmax=VMAX_BG,  
                                s=1.0, alpha=0.9, marker='s', transform=ccrs.PlateCarree(), zorder=1)
        if sm_bg is None: sm_bg = scatter_bg
        
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black', zorder=5)
        
        if os.path.exists(CITY_SHP_PATH):
            try:
                city_geom = cfeature.ShapelyFeature(shpreader.Reader(CITY_SHP_PATH).geometries(),
                                                    ccrs.PlateCarree(), facecolor='none', 
                                                    edgecolor='dimgray', linewidth=0.4, linestyle='--', alpha=0.7)
                ax.add_feature(city_geom, zorder=3)
            except Exception as e:
                pass

        if os.path.exists(PROVINCE_SHP_PATH):
            try:
                prov_geom = cfeature.ShapelyFeature(shpreader.Reader(PROVINCE_SHP_PATH).geometries(),
                                                    ccrs.PlateCarree(), facecolor='none', 
                                                    edgecolor='#222222', linewidth=0.9, linestyle='-', alpha=1.0)
                ax.add_feature(prov_geom, zorder=4)
            except Exception as e:
                pass
        else:
            try:
                provinces = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
                ax.add_feature(provinces, edgecolor='black', linewidth=0.8, zorder=4)
            except: pass

        ocean_mask = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='white')
        ax.add_feature(ocean_mask, edgecolor='none', zorder=5)
                                 
        if not site_monthly_stats.empty:
            # 散点颜色映射为 NMB 值
            scatter_err = ax.scatter(site_monthly_stats['lon'].values, site_monthly_stats['lat'].values, 
                                     c=site_monthly_stats['NMB'].values, 
                                     cmap=cmap_err, vmin=-V_ERR_NMB, vmax=V_ERR_NMB, 
                                     s=60, alpha=1.0, edgecolors='black', linewidths=1.0, 
                                     transform=ccrs.PlateCarree(), zorder=6)
            if sm_err is None: sm_err = scatter_err
                
        if not missing_sites_df.empty:
            ax.scatter(missing_sites_df['lon'].values, missing_sites_df['lat'].values, 
                       c='lightgray', marker='X', s=50, edgecolors='dimgray', linewidths=0.8, 
                       transform=ccrs.PlateCarree(), zorder=6)
        
        valid_count = len(valid_site_codes)
        ax.set_title(f'{target_month}月 (n={valid_count})', fontproperties=my_font, fontsize=24, weight='bold', pad=12)
        
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        if idx < 9:  gl.bottom_labels = False   
        if idx % 3 != 0: gl.left_labels = False 
        gl.xlabel_style = {'size': 14, 'color': 'black', 'weight': 'bold'}
        gl.ylabel_style = {'size': 14, 'color': 'black', 'weight': 'bold'}

    # =============================================================================
    # 4. 生成全局共享双色标系统与布局精调
    # =============================================================================
    print("\n -> [3/3] 正在生成全局统一色标并输出最终大图...")
    
    plt.subplots_adjust(left=0.06, right=0.88, bottom=0.16, top=0.90, wspace=0.08, hspace=0.12)
    
    if sm_bg is not None:
        cbar_ax1 = fig.add_axes([0.15, 0.06, 0.45, 0.015]) 
        cbar1 = fig.colorbar(sm_bg, cax=cbar_ax1, orientation='horizontal')
        cbar1.set_label(r'【背景底图】长三角 1km 预测月均 PM$_{2.5}$ 浓度 ($\mu g/m^3$)', fontproperties=my_font, fontsize=24, weight='bold')
        cbar1.ax.tick_params(labelsize=18)

    if sm_err is not None:
        # 右侧色标底端对齐抬高后的主图 (0.16)
        cbar_ax2 = fig.add_axes([0.91, 0.16, 0.015, 0.7])
        cbar2 = fig.colorbar(sm_err, cax=cbar_ax2, orientation='vertical')
        cbar2.set_label(r'【圆圈站点】独立测试站点 NMB 偏差 (%)', fontproperties=my_font, fontsize=24, weight='bold')
        cbar2.ax.tick_params(labelsize=18)

    valid_marker = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='white', 
                                 markeredgecolor='black', markersize=20, label='有效验证站点')
    missing_marker = mlines.Line2D([], [], color='white', marker='X', markerfacecolor='lightgray', 
                                   markeredgecolor='dimgray', markersize=20, label='数据缺失站点')
    
    handles_list = [valid_marker, missing_marker]
    
    if os.path.exists(PROVINCE_SHP_PATH):
        prov_line = mlines.Line2D([], [], color='#222222', linestyle='-', linewidth=4.0, label='省级边界')
        handles_list.append(prov_line)
        
    if os.path.exists(CITY_SHP_PATH):
        city_line = mlines.Line2D([], [], color='dimgray', linestyle='--', linewidth=3.0, label='地级市边界')
        handles_list.append(city_line)

    legend_font = FontProperties(fname=FONT_PATH, size=24) if os.path.exists(FONT_PATH) else FontProperties(size=24)

    fig.legend(handles=handles_list, loc='center right', 
               bbox_to_anchor=(0.90, 0.08), prop=legend_font, 
               frameon=True, framealpha=1.0, edgecolor='black', facecolor='whitesmoke',
               borderpad=0.8, labelspacing=0.8, handletextpad=0.6)

    save_path = os.path.join(FIG_DIR, "Spatial_Monthly_NMB_2025_12Months.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print("="*70)
    print(f"🎉 恭喜！12个月份的 NMB 标准化平均偏差全矩阵大组图已生成！\n👉 图片保存至: {save_path}")
    print("="*70)

if __name__ == "__main__":
    main()