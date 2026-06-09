import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, train_test_split
from sklearn.cluster import KMeans  
from sklearn.inspection import permutation_importance
import warnings
import chinese_calendar as conc
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
from cuml.ensemble import RandomForestRegressor
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import joblib

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 基础配置
# =============================================================================
DATA_FILE = "/home/wangzonghan/bisheshuju/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"
OUTPUT_DIR = "/home/wangzonghan/bisheshuju/Results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FONT_PATH = "/home/wangzonghan/bisheshuju/fonts/SimHei.ttf"
my_font = FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else FontProperties()
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 1. 数据加载与特征池构建
# =============================================================================
def load_and_prepare_data():
    print("\n" + "="*70)
    print("📂 [1/5] 加载数据集并构建全量特征池")
    print("="*70)
    
    df = pd.read_parquet(DATA_FILE)
    
    if 'month' not in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['season'] = (df['date'].dt.month % 12 // 3 + 1)
        df['is_holiday'] = df['date'].apply(lambda x: 1 if conc.is_holiday(x) else 0)
            
    exclude_cols = ['date', 'datetime', 'site_code', 'pm25_hourly', 'lon', 'lat']
    all_candidate_features = [c for c in df.columns if c not in exclude_cols]
    
    print(f"  -> 共加载 {len(all_candidate_features)} 个候选特征 (已成功包含 'hour')。")
    return df, all_candidate_features

# =============================================================================
# 2. 空间分层抽样 (K-Means)
# =============================================================================
def spatial_stratified_split(df):
    print("\n" + "="*70)
    print("🌍 [2/5] 执行空间分层抽样 (K-Means)")
    print("="*70)
    
    sites_info = df[['site_code', 'lon', 'lat']].drop_duplicates().reset_index(drop=True)
    num_test = int(len(sites_info) * 0.15)
    
    kmeans = KMeans(n_clusters=num_test, random_state=42)
    sites_info['spatial_cluster'] = kmeans.fit_predict(sites_info[['lon', 'lat']])
    
    test_sites_df = sites_info.groupby('spatial_cluster').apply(
        lambda x: x.sample(n=1, random_state=42)
    ).reset_index(drop=True)
    
    test_sites = test_sites_df['site_code'].tolist()
    train_sites = [s for s in sites_info['site_code'] if s not in test_sites]
    
    train_df = df[df['site_code'].isin(train_sites)].copy()
    test_df = df[df['site_code'].isin(test_sites)].copy()
    
    print(f"  -> 训练/验证集分配: {len(train_sites)} 个站点, {len(train_df):,} 个样本")
    print(f"  -> 独立测试集分配: {len(test_sites)} 个站点, {len(test_df):,} 个样本")
    
    return train_df, test_df

# =============================================================================
# 3. 绘制站点空间分布图 
# =============================================================================
def plot_spatial_distribution(train_df, test_df):
    print("\n" + "="*70)
    print("🗺️ [3/5] 生成站点空间分布可视化")
    print("="*70)
    
    GEBCO_PATH = "/home/wangzonghan/bisheshuju/GEBCO/GEBCO_2025_YRD_1km_with_slope.nc" 
    train_loc = train_df[['site_code', 'lon', 'lat']].drop_duplicates()
    test_loc = test_df[['site_code', 'lon', 'lat']].drop_duplicates()
    
    fig = plt.figure(figsize=(13, 10), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    min_lon, max_lon = 114.5, 123
    min_lat, max_lat = 27.0, 35.5

    ds = xr.open_dataset(GEBCO_PATH)
    elev_yrd = ds['DEM'].sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    if len(elev_yrd.lat) == 0:
        elev_yrd = ds['DEM'].sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))

    lon_grid, lat_grid = np.meshgrid(elev_yrd.lon, elev_yrd.lat)
    mesh = ax.pcolormesh(lon_grid, lat_grid, elev_yrd.values, 
                         cmap='terrain', vmin=-500, vmax=1500, 
                         transform=ccrs.PlateCarree(), shading='auto', zorder=1)
    
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.06, aspect=40)
    cbar.set_label('陆地海拔高度 (m) / 海洋深度 (<0m)', fontproperties=my_font, fontsize=12)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black', zorder=2)
    try:
        provinces = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none')
        ax.add_feature(provinces, edgecolor='black', linewidth=0.8, zorder=3)
    except Exception:
        pass
    
    gl = ax.gridlines(draw_labels=True, linewidth=0, color='none') 
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator([115, 116, 117, 118, 119, 120, 121, 122])
    gl.ylocator = mticker.FixedLocator([28.5, 30.0, 31.5, 33.0, 34.5])
    gl.xlabel_style = {'size': 13, 'weight': 'bold', 'color': 'black'}
    gl.ylabel_style = {'size': 13, 'weight': 'bold', 'color': 'black'}
    
    rect = Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat, 
                     linewidth=3, edgecolor='red', facecolor='none', 
                     transform=ccrs.PlateCarree(), zorder=4)
    ax.add_patch(rect)

    ax.scatter(train_loc['lon'], train_loc['lat'], c='darkgray', s=50, alpha=0.9, 
                edgecolors='black', linewidths=0.8, transform=ccrs.PlateCarree(), zorder=5)
    ax.scatter(test_loc['lon'], test_loc['lat'], c='red', s=220, marker='*', 
                edgecolors='darkred', linewidths=1.2, transform=ccrs.PlateCarree(), zorder=6)

    text_str = "1km 网格统计:\nX(经度) = 851 个\nY(纬度) = 851 个"
    props = dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='black', alpha=1.0)
    ax.text(1.04, 0.38, text_str, transform=ax.transAxes, fontsize=12, fontproperties=my_font,
            verticalalignment='bottom', horizontalalignment='left', bbox=props, zorder=10)

    red_box_line = mlines.Line2D([], [], color='red', linewidth=3, label='1km 网格边界')
    train_marker = mlines.Line2D([], [], color='none', marker='o', markerfacecolor='darkgray',
                                 markeredgecolor='black', markersize=8, label='建模站点\n(Training & Validation)')
    test_marker = mlines.Line2D([], [], color='none', marker='*', markerfacecolor='red',
                                markeredgecolor='darkred', markersize=14, label='独立测试站点\n(Hold-out Test)')

    leg = ax.legend(handles=[red_box_line, train_marker, test_marker], loc='lower left',
                    bbox_to_anchor=(1.04, 0.05), prop=my_font, framealpha=1.0, 
                    edgecolor='black', borderaxespad=0., labelspacing=1.0)
    leg.set_zorder(10)

    ax.set_title('长三角地区 PM2.5 (小时级) 预测模型站点空间隔离分布', fontproperties=my_font, fontsize=18, pad=15)

    fig_dir = os.path.join(OUTPUT_DIR, "Figures_随机森林")
    os.makedirs(fig_dir, exist_ok=True)
    save_path = os.path.join(fig_dir, "Spatial_Sites_Distribution_GEBCO.png")
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 站点分布隔离证明图已保存至: {save_path}")

# =============================================================================
# 4. 全局重要性阈值特征筛选 (Permutation Importance)
# =============================================================================
def threshold_based_feature_selection(train_df, all_candidate_features):
    print("\n" + "="*70)
    print("🔍 [4/5] 执行全局重要性阈值特征筛选")
    print("="*70)
    
    sfs_train_df, sfs_val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    y_train = sfs_train_df['pm25_hourly'].astype('float32').values
    y_val = sfs_val_df['pm25_hourly'].astype('float32').values
    
    X_train_full = sfs_train_df[all_candidate_features].astype('float32').values
    X_val_full = sfs_val_df[all_candidate_features].astype('float32').values
    
    eval_model = RandomForestRegressor(n_estimators=100, max_depth=30, random_state=42)
    eval_model.fit(X_train_full, y_train)
    
    base_score = eval_model.score(X_val_full, y_val)
    print(f"  -> 初始全量特征模型验证集 R²: {base_score:.4f}")
    
    print("  -> 正在进行 Permutation 测试评估特征重要性...")
    result = permutation_importance(eval_model, X_val_full, y_val, n_repeats=5, random_state=42, scoring='r2', n_jobs=1)
    
    threshold = 0.001
    best_features = []
    
    print(f"\n  -> 特征真实贡献度评估 (阈值: > {threshold}):")
    for i, feature in enumerate(all_candidate_features):
        imp = result.importances_mean[i]
        if imp > threshold:
            best_features.append(feature)
        else:
            print(f"      ❌ 剔除: {feature:<18} (Importance: {imp:.4f})")
            
    print(f"\n  🎯 筛选结束，保留特征数: {len(best_features)}")
    return best_features

# =============================================================================
# 5. 随机搜索调优与独立测试评估 
# =============================================================================
def hyperparameter_tuning_and_evaluation(train_df, test_df, selected_features):
    print("\n" + "="*70)
    print("⚙️ [5/5] 执行【随机搜索】调优及【双尺度独立测试检验】")
    print("="*70)
    
    X_train = train_df[selected_features].astype('float32').values
    y_train = train_df['pm25_hourly'].astype('float32').values
    X_test = test_df[selected_features].astype('float32').values
    
    param_dist = {
        'n_estimators': [100, 200],          
        'max_depth': [15, 18, 20, 22],                
        'max_features': [0.4, 0.6],            
        'min_samples_split': [2],                  
        'min_samples_leaf': [1],                    
        'bootstrap': [True]
    }
    
    rf = RandomForestRegressor(
        n_bins=128,                       
        random_state=42
    )
    
    random_search = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=KFold(n_splits=3, shuffle=True, random_state=42), 
        scoring='r2', 
        n_jobs=1, 
        verbose=2,
        random_state=42,
        error_score=0 
    )
    
    print("  ⏳ 启动随机参数寻优搜索 (共需拟合 30 次，请耐心等待...)")
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    print(f"\n  🏆 最优参数组合: {random_search.best_params_}")
    
    # 获取小时级预测值
    preds_hourly = best_model.predict(X_test)
    if hasattr(preds_hourly, 'to_numpy'): preds_hourly = preds_hourly.to_numpy()
        
    # ==========================================
    # 🎯 检验 1：小时尺度 (Hourly Scale) 评估
    # ==========================================
    y_test_hourly = test_df['pm25_hourly'].astype('float32').values
    r2_hourly = r2_score(y_test_hourly, preds_hourly)
    rmse_hourly = np.sqrt(mean_squared_error(y_test_hourly, preds_hourly))
    
    # ==========================================
    # 🎯 检验 2：日均尺度 (Daily Scale) 聚合评估
    # ==========================================
    # 将预测结果贴回 test_df 副本中
    eval_df = test_df[['date', 'site_code', 'pm25_hourly']].copy()
    eval_df['pred_hourly'] = preds_hourly
    
    # 按天和站点 groupby 求均值，完美模拟真实的日均聚合逻辑！
    daily_eval_df = eval_df.groupby(['date', 'site_code']).mean().reset_index()
    y_test_daily = daily_eval_df['pm25_hourly'].values
    preds_daily = daily_eval_df['pred_hourly'].values
    
    r2_daily = r2_score(y_test_daily, preds_daily)
    rmse_daily = np.sqrt(mean_squared_error(y_test_daily, preds_daily))
    
    print("\n" + "🔥"*30)
    print("  🎯 模型独立测试站点终极评估报告 (Hold-out Sites)")
    print("🔥"*30)
    print(f"  🕒 【检验一：小时级浓度捕捉能力】")
    print(f"      -> R²   : {r2_hourly:.4f}")
    print(f"      -> RMSE : {rmse_hourly:.2f} μg/m³")
    print(f"  📅 【检验二：日均级宏观解释能力】")
    print(f"      -> R²   : {r2_daily:.4f}")
    print(f"      -> RMSE : {rmse_daily:.2f} μg/m³")
    print("="*60)
    
    return best_model

# =============================================================================
# 6. 特征重要性可视化
# =============================================================================
def extract_and_plot_feature_importance(best_model, feature_names, test_df):
    print("\n" + "="*70)
    print("📊 生成模型特征重要性分析图")
    print("="*70)
    
    X_test = test_df[feature_names].astype('float32').values
    y_test = test_df['pm25_hourly'].astype('float32').values # 🌟 对应小时级
    
    result = permutation_importance(
        best_model, X_test, y_test, n_repeats=5, random_state=42, scoring='r2', n_jobs=1
    )
    
    importances = result.importances_mean
        
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n  -> 🌟 变量重要性排名 (对独立测试集 R² 的真实贡献):")
    for index, row in df_imp.iterrows():
        print(f"      {row['Feature']:<18}: {row['Importance']:.4f}")
    
    plt.figure(figsize=(12, 10), dpi=300)
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='magma')
    plt.title('最优模型特征重要性评估 (Permutation - 小时级模型)', fontproperties=my_font, fontsize=16)
    plt.xlabel(r'特征对验证集 $R^2$ 的贡献度', fontproperties=my_font, fontsize=14)
    plt.ylabel('入模特征变量', fontproperties=my_font, fontsize=14)
    plt.tight_layout()
    
    fig_dir = os.path.join(OUTPUT_DIR, "Figures_随机森林")
    os.makedirs(fig_dir, exist_ok=True)
    save_path = os.path.join(fig_dir, "RF_Feature_Importance_Permutation_Terrain.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  ✅ 特征重要性排序图已保存至: {save_path}")

# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == "__main__":
    df, all_candidate_features = load_and_prepare_data()
    train_df, test_df = spatial_stratified_split(df)
    plot_spatial_distribution(train_df, test_df)
    
    best_features = threshold_based_feature_selection(train_df, all_candidate_features)
    best_rf_model = hyperparameter_tuning_and_evaluation(train_df, test_df, best_features)
    extract_and_plot_feature_importance(best_rf_model, best_features, test_df)
    
    model_dir = os.path.join(OUTPUT_DIR, "Models_随机森林")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_rf_model_terrain.pkl")
    features_path = os.path.join(model_dir, "best_features_list_terrain.pkl")
    joblib.dump(best_rf_model, model_path)
    joblib.dump(best_features, features_path)
    print(f"\n🎉 核心随机森林模型及特征列表已保存。")