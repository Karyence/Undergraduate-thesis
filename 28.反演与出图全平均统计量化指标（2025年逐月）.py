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
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
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

TARGET_MONTHS = range(1, 13)

# --- IDW 空间插值函数 ---
def idw_interpolation(x_known, y_known, values_known, x_grid, y_grid, k=15, p=2):
    tree = cKDTree(np.column_stack((x_known, y_known)))
    dist, idx = tree.query(np.column_stack((x_grid, y_grid)), k=k)
    weights = 1.0 / (dist ** p + 1e-12)
    interpolated = np.sum(weights * values_known[idx], axis=1) / np.sum(weights, axis=1)
    return interpolated

# --- 误差量化指标函数 ---
def calc_metrics(obs, pred):
    valid_mask = ~np.isnan(obs) & ~np.isnan(pred)
    obs = obs[valid_mask]
    pred = pred[valid_mask]
    if len(obs) < 15: return np.nan, np.nan, np.nan, np.nan, np.nan
    nmb = np.sum(pred - obs) / np.sum(obs) * 100
    nme = np.sum(np.abs(pred - obs)) / np.sum(obs) * 100
    if np.std(obs) > 1e-6 and np.std(pred) > 1e-6:
        r2 = r2_score(obs, pred)
        coeff = np.corrcoef(obs, pred)[0, 1]
    else: r2, coeff = np.nan, np.nan
    rmse = np.sqrt(mean_squared_error(obs, pred))
    return nmb, nme, r2, coeff, rmse

def main():
    print("="*80)
    print("🚀 启动：长三角 1km 时空降维 + IDW残差空间校正")
    print("="*80)

    # =============================================================================
    # 1. 加载基础真值数据并切分站点
    # =============================================================================
    print(" 📂 [1/4] 正在加载日均真值数据并切分 15% 独立测试集...")
    original_df = pd.read_parquet(DATASET_PATH)
    original_df['date'] = pd.to_datetime(original_df['date'])
    original_df['month'] = original_df['date'].dt.month
    original_df['date_str'] = original_df['date'].dt.strftime('%Y%m%d')
    
    # 将含有 12 个月的小时级真值，在内存里压缩成日均真值，供下游 IDW 和指标计算使用
    original_df = original_df.groupby(['site_code', 'lon', 'lat', 'month', 'date_str'])['pm25_hourly'].mean().reset_index()
    original_df.rename(columns={'pm25_hourly': 'pm25_daily'}, inplace=True)
    
    sites_info = original_df[['site_code', 'lon', 'lat']].drop_duplicates().reset_index(drop=True)
    num_test = int(len(sites_info) * 0.15)
    kmeans = KMeans(n_clusters=num_test, random_state=42, n_init=10)
    sites_info['spatial_cluster'] = kmeans.fit_predict(sites_info[['lon', 'lat']])
    
    test_sites_df = sites_info.groupby('spatial_cluster').apply(lambda x: x.sample(n=1, random_state=42)).reset_index(drop=True)
    test_sites_list = test_sites_df['site_code'].tolist()
    test_sites_coords = test_sites_df[['site_code', 'lon', 'lat']].copy() 
    
    test_df = original_df[original_df['site_code'].isin(test_sites_list)].copy()
    train_df = original_df[~original_df['site_code'].isin(test_sites_list)].copy()

    # =============================================================================
    # 2. 唤醒模型
    # =============================================================================
    print(" ⏳ [2/4] 正在唤醒 RF 模型与特征清单...")
    best_rf_model = joblib.load(os.path.join(MODEL_DIR, "best_rf_model_terrain.pkl"))
    best_features = joblib.load(os.path.join(MODEL_DIR, "best_features_list_terrain.pkl"))

    print("\n" + "="*80)
    print("🔥 [3/4] 引擎点火！执行：小时级提取 -> 日均折叠 -> IDW校正 -> 月度聚合")
    print("="*80)
    
    fig = plt.figure(figsize=(20, 26), dpi=300) 
    fig.suptitle('长三角 1km 混合模型 (RF+IDW) 空间校正演变图 (2025年1-12月)', 
                 fontproperties=my_font, fontsize=36, weight='bold', y=0.94)
                 
    sm_bg = None; sm_scatter = None

    for idx, target_month in enumerate(TARGET_MONTHS):
        print(f"\n ⏳ [渲染面板 {idx+1}/12]: 正在处理 {target_month:02d} 月...")
        month_str = f"2025{target_month:02d}"
        
        # 读取 Hourly 超级特征矩阵
        grid_files_pattern = f"{INFERENCE_DIR}/YRD_Grid_Hourly_Features_Terrain_{month_str}*.parquet"
        grid_files = glob.glob(grid_files_pattern)
        if not grid_files: 
            print(f"   ⚠️ 跳过: 缺失 {month_str} 的网格特征文件。")
            continue
            
        monthly_pred_sum = None
        pixel_valid_counts = None
        grid_lon_lat = None
        daily_records = [] 
        
        test_month_df = test_df[test_df['month'] == target_month].copy()
        
        for daily_file in grid_files:
            current_date_str = os.path.basename(daily_file).split('_')[-1].split('.')[0]
            
            # 1. 读取单日小时级数据 (1700万行)
            grid_df = pd.read_parquet(daily_file)
            X_grid = grid_df[best_features].astype('float32')
            
            # 2. 全场小时预测
            grid_df['pred_hourly'] = best_rf_model.predict(X_grid)
            
            # 3. 空间折叠降维为日均场 (72万行)
            daily_grid = grid_df.groupby(['lon', 'lat'])['pred_hourly'].mean().reset_index()
            daily_grid = daily_grid.sort_values(by=['lat', 'lon']).reset_index(drop=True)
            
            if grid_lon_lat is None: grid_lon_lat = daily_grid[['lon', 'lat']].copy()
            preds = daily_grid['pred_hourly'].values
            
            # 4. IDW 残差空间校正 (在日均维度上进行)
            today_train_obs = train_df[train_df['date_str'] == current_date_str]
            if not today_train_obs.empty and len(today_train_obs) > 5:
                grid_tree = cKDTree(daily_grid[['lon', 'lat']].values)
                _, train_indices = grid_tree.query(today_train_obs[['lon', 'lat']].values)
                
                model_preds_at_train = preds[train_indices]
                residuals = today_train_obs['pm25_daily'].values - model_preds_at_train
                
                # 计算全场残差并补偿
                residual_grid = idw_interpolation(
                    today_train_obs['lon'].values, today_train_obs['lat'].values, residuals, 
                    daily_grid['lon'].values, daily_grid['lat'].values
                )
                preds = np.clip(preds + residual_grid, 0, 300)
            
            # 5. 累加至当月均值
            if monthly_pred_sum is None: 
                monthly_pred_sum = np.zeros(len(daily_grid), dtype=float)
                pixel_valid_counts = np.zeros(len(daily_grid), dtype=float)
                
            monthly_pred_sum += preds
            pixel_valid_counts += 1
            
            # 6. 提取独立测试集预测值用于验证
            tree = cKDTree(daily_grid[['lon', 'lat']].values)
            today_obs_df = test_month_df[test_month_df['date_str'] == current_date_str].copy()
            if not today_obs_df.empty:
                _, indices = tree.query(today_obs_df[['lon', 'lat']].values)
                today_obs_df['daily_extracted_pred'] = preds[indices]
                daily_records.append(today_obs_df[['site_code', 'pm25_daily', 'daily_extracted_pred']])
                
            # 7. 极限内存回收
            del grid_df, X_grid, daily_grid
            gc.collect()
        
        # 计算该月的最终网格均值
        with np.errstate(divide='ignore', invalid='ignore'):
            grid_monthly_mean = np.where(pixel_valid_counts > 0, 
                                         monthly_pred_sum / pixel_valid_counts, 
                                         np.nan)
        grid_lon_lat['monthly_pm25'] = np.clip(grid_monthly_mean, 0, 150)

        # 统计该月的综合验证指标
        if daily_records:
            all_daily_df = pd.concat(daily_records)
            site_metrics = []
            for site, group in all_daily_df.groupby('site_code'):
                obs = group['pm25_daily'].values
                pred = group['daily_extracted_pred'].values
                nmb, nme, r2, coeff, rmse = calc_metrics(obs, pred)
                if not np.isnan(r2) and r2 >= 0: site_metrics.append([nmb, nme, r2, coeff, rmse])
            if site_metrics:
                site_metrics_arr = np.array(site_metrics)
                m_nmb, m_nme, m_r2, m_coeff, m_rmse = np.nanmean(site_metrics_arr, axis=0)
            else: m_nmb = m_nme = m_r2 = m_coeff = m_rmse = np.nan
        else: m_nmb = m_nme = m_r2 = m_coeff = m_rmse = np.nan

        site_stats = test_month_df.groupby(['site_code', 'lon', 'lat']).agg(obs_mean=('pm25_daily', 'mean')).reset_index().dropna(subset=['obs_mean']) if not test_month_df.empty else pd.DataFrame()
        valid_site_codes = site_stats['site_code'].tolist() if not site_stats.empty else []
        missing_sites_df = test_sites_coords[~test_sites_coords['site_code'].isin(valid_site_codes)]

        # =============================================================================
        # 4. 制图面板 
        # =============================================================================
        ax = fig.add_subplot(4, 3, idx + 1, projection=ccrs.PlateCarree())
        ax.set_extent([114.5, 123.0, 27.0, 35.5], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='white', zorder=0)
        
        lons_perfect = np.linspace(114.5, 123.0, 851)
        lats_perfect = np.linspace(27.0, 35.5, 851)
        lon_grid_perfect, lat_grid_perfect = np.meshgrid(lons_perfect, lats_perfect)

        points = (grid_lon_lat['lon'].values, grid_lon_lat['lat'].values)
        values = grid_lon_lat['monthly_pm25'].values
        
        pm25_nearest = griddata(points, values, (lon_grid_perfect, lat_grid_perfect), method='nearest')
        pm25_linear = griddata(points, values, (lon_grid_perfect, lat_grid_perfect), method='linear')
        pm25_smooth = np.where(np.isnan(pm25_linear), pm25_nearest, pm25_linear)

        levels = np.linspace(10, 100, 91) 
        mesh_bg = ax.contourf(lon_grid_perfect, lat_grid_perfect, pm25_smooth, 
                              levels=levels, cmap='Spectral_r', extend='both',
                              transform=ccrs.PlateCarree(), zorder=1, alpha=0.9)
        if sm_bg is None: sm_bg = mesh_bg
                                 
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='white'), edgecolor='none', zorder=2)
                                 
        if not site_stats.empty:
            sc = ax.scatter(site_stats['lon'].values, site_stats['lat'].values, c=site_stats['obs_mean'].values, cmap='Spectral_r', vmin=10, vmax=100, s=70, alpha=1.0, edgecolors='black', linewidths=1.2, zorder=4)
            if sm_scatter is None: sm_scatter = sc
                
        stats_text = f"R²: {m_r2:.2f}\nCoeff: {m_coeff:.2f}\nRMSE: {m_rmse:.1f}\nNMB: {m_nmb:+.1f}%\nNME: {m_nme:.1f}%" if not np.isnan(m_r2) else "真值暂缺"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=16, weight='bold', 
                verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=1.5), zorder=6)
        
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black', zorder=5)
        
        if os.path.exists(CITY_SHP_PATH):
            try:
                city_geom = cfeature.ShapelyFeature(shpreader.Reader(CITY_SHP_PATH).geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='dimgray', linewidth=0.4, linestyle='--', alpha=0.7)
                ax.add_feature(city_geom, zorder=3)
            except: pass
        if os.path.exists(PROVINCE_SHP_PATH):
            try:
                prov_geom = cfeature.ShapelyFeature(shpreader.Reader(PROVINCE_SHP_PATH).geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='#222222', linewidth=0.9, linestyle='-', alpha=1.0)
                ax.add_feature(prov_geom, zorder=4)
            except: pass
            
        if not missing_sites_df.empty:
            ax.scatter(missing_sites_df['lon'].values, missing_sites_df['lat'].values, 
                       c='lightgray', marker='X', s=50, edgecolors='dimgray', linewidths=0.8, 
                       transform=ccrs.PlateCarree(), zorder=6)
        
        ax.set_title(f'{target_month}月', fontproperties=my_font, fontsize=24, weight='bold', pad=12)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        if idx < 9: gl.bottom_labels = False
        if idx % 3 != 0: gl.left_labels = False
        gl.xlabel_style = {'size': 14, 'color': 'black', 'weight': 'bold'}
        gl.ylabel_style = {'size': 14, 'color': 'black', 'weight': 'bold'}

    # =============================================================================
    # 5. 色标配置与保存 
    # =============================================================================
    print("\n -> [4/4] 正在生成全局统一色标并输出最终大图...")
    
    plt.subplots_adjust(left=0.06, right=0.88, bottom=0.16, top=0.90, wspace=0.08, hspace=0.12)
    
    if sm_bg is not None:
        cbar_ax1 = fig.add_axes([0.15, 0.06, 0.45, 0.015])
        cbar1 = fig.colorbar(sm_bg, cax=cbar_ax1, orientation='horizontal')
        cbar1.set_label(r'【背景底图】长三角 1km 混合反演预测月均 PM$_{2.5}$ 浓度 ($\mu g/m^3$)', fontproperties=my_font, fontsize=24, weight='bold')
        cbar1.ax.tick_params(labelsize=18)
    
    if sm_scatter is not None:
        cbar_ax2 = fig.add_axes([0.91, 0.16, 0.015, 0.7])
        cbar2 = fig.colorbar(sm_scatter, cax=cbar_ax2, orientation='vertical')
        cbar2.set_label(r'【圆圈站点】独立测试站点真实观测 PM$_{2.5}$ 浓度 ($\mu g/m^3$)', fontproperties=my_font, fontsize=24, weight='bold')
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

    save_path = f"{FIG_DIR}/YRD_PM25_Monthly_Hybrid_RF_IDW_2025.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print("="*80)
    print("🎉 极度丝滑的等值线填充大图渲染完毕！")
    print(f"👉 终极完美图路径: {save_path}")
    print("="*80)

if __name__ == "__main__":
    main()
