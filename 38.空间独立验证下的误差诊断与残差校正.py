import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from scipy.spatial import cKDTree
import chinese_calendar as conc  
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 基础配置与路径
# =============================================================================
BASE_DIR = "/home/wangzonghan/bisheshuju"
MODEL_DIR = f"{BASE_DIR}/Results/Models_随机森林"
DATASET_PATH = f"{BASE_DIR}/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet" 
OUTPUT_DIR = f"{BASE_DIR}/Results/Figures_残差诊断"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FONT_PATH = f"{BASE_DIR}/fonts/SimHei.ttf"
my_font = FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else FontProperties()
plt.rcParams['axes.unicode_minus'] = False

# --- IDW 空间插值函数 ---
def idw_interpolation(x_known, y_known, values_known, x_grid, y_grid, k=15, p=2):
    if len(x_known) == 0: return np.zeros(len(x_grid))
    k = min(k, len(x_known))
    tree = cKDTree(np.column_stack((x_known, y_known)))
    dist, idx = tree.query(np.column_stack((x_grid, y_grid)), k=k)
    weights = 1.0 / (dist ** p + 1e-12)
    interpolated = np.sum(weights * values_known[idx], axis=1) / np.sum(weights, axis=1)
    return interpolated

def calc_rmse(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))

def main():
    print("="*80)
    print("🚀 启动： RF vs RF-IDW 多区域无泄露诊断验证")
    print("="*80)

    # -------------------------------------------------------------------------
    # 1. 加载数据与空间隔离
    # -------------------------------------------------------------------------
    print(" 📂 [1/4] 加载数据并补全时间特征...")
    df_hourly = pd.read_parquet(DATASET_PATH)
    df_hourly['date'] = pd.to_datetime(df_hourly['date'])
    df_hourly['date_str'] = df_hourly['date'].dt.strftime('%Y%m%d')
    
    if 'month' not in df_hourly.columns:
        print("   -> 正在动态解析时间特征 (月份、星期、节假日等)...")
        df_hourly['month'] = df_hourly['date'].dt.month
        df_hourly['day_of_week'] = df_hourly['date'].dt.dayofweek
        df_hourly['is_weekend'] = df_hourly['day_of_week'].isin([5, 6]).astype(int)
        df_hourly['season'] = (df_hourly['month'] % 12 // 3 + 1).astype(int)
        df_hourly['is_holiday'] = df_hourly['date'].apply(lambda x: 1 if conc.is_holiday(x) else 0)
    
    # 严格空间隔离
    print("   -> 正在执行 K-Means 空间隔离...")
    sites_info = df_hourly[['site_code', 'lon', 'lat']].drop_duplicates().reset_index(drop=True)
    num_test = int(len(sites_info) * 0.15)
    kmeans = KMeans(n_clusters=num_test, random_state=42, n_init=10)
    sites_info['spatial_cluster'] = kmeans.fit_predict(sites_info[['lon', 'lat']])
    
    test_sites_df = sites_info.groupby('spatial_cluster').apply(lambda x: x.sample(n=1, random_state=42)).reset_index(drop=True)
    test_sites_list = test_sites_df['site_code'].tolist()

    # -------------------------------------------------------------------------
    # 2. 调用 RF 模型执行全量站点预测
    # -------------------------------------------------------------------------
    print(" 🧠 [2/4] 调用主模型预测...")
    model = joblib.load(os.path.join(MODEL_DIR, "best_rf_model_terrain.pkl"))
    best_features = joblib.load(os.path.join(MODEL_DIR, "best_features_list_terrain.pkl"))
    
    # 预测
    df_hourly['pred_hourly'] = model.predict(df_hourly[best_features].astype('float32'))
    
    # -------------------------------------------------------------------------
    # 3. 聚合至日均尺度并执行无泄露 IDW 补偿
    # -------------------------------------------------------------------------
    print(" 🔄 [3/4] 聚合至日尺度并执行无泄露 IDW 补偿...")
    agg_dict = {'pm25_hourly': 'mean', 'pred_hourly': 'mean'}
    if 'DEM' in df_hourly.columns: agg_dict['DEM'] = 'first' # 保留 DEM 用于后续切分
    
    df_daily = df_hourly.groupby(['date_str', 'site_code', 'lon', 'lat']).agg(agg_dict).reset_index()
    df_daily.rename(columns={'pm25_hourly': 'pm25_daily', 'pred_hourly': 'pred_rf_daily'}, inplace=True)

    # 划分训练集和测试集
    df_train = df_daily[~df_daily['site_code'].isin(test_sites_list)].copy()
    df_test = df_daily[df_daily['site_code'].isin(test_sites_list)].copy()
    
    # 初始化
    df_test['pred_idw_daily'] = df_test['pred_rf_daily']

    # 逐日执行残差插值 (严格防泄露：仅用 train 残差补充 test)
    for date_str, test_group in df_test.groupby('date_str'):
        train_group = df_train[df_train['date_str'] == date_str]
        if len(train_group) < 5: continue
        
        train_lon, train_lat = train_group['lon'].values, train_group['lat'].values
        train_res = train_group['pm25_daily'].values - train_group['pred_rf_daily'].values
        
        test_lon, test_lat = test_group['lon'].values, test_group['lat'].values
        
        interp_res = idw_interpolation(train_lon, train_lat, train_res, test_lon, test_lat, k=15, p=2)
        df_test.loc[test_group.index, 'pred_idw_daily'] = test_group['pred_rf_daily'] + interp_res

    df_test['pred_idw_daily'] = np.clip(df_test['pred_idw_daily'], 0, 300)

    # -------------------------------------------------------------------------
    # 4. 多区域验证与成图
    # -------------------------------------------------------------------------
    print(" 📊 [4/4] 计算各区域评估指标并出图...")
    
    # 1. 全局
    rmse_all_rf = calc_rmse(df_test['pm25_daily'], df_test['pred_rf_daily'])
    rmse_all_idw = calc_rmse(df_test['pm25_daily'], df_test['pred_idw_daily'])
    
    # 2. 山区 (DEM >= 50m)
    df_mt = df_test[df_test['DEM'] >= 50] if 'DEM' in df_test.columns else df_test
    rmse_mt_rf = calc_rmse(df_mt['pm25_daily'], df_mt['pred_rf_daily'])
    rmse_mt_idw = calc_rmse(df_mt['pm25_daily'], df_mt['pred_idw_daily'])
    
    # 3. 沿海 (经度 >= 121.0)
    df_coast = df_test[df_test['lon'] >= 121.0]
    rmse_coast_rf = calc_rmse(df_coast['pm25_daily'], df_coast['pred_rf_daily'])
    rmse_coast_idw = calc_rmse(df_coast['pm25_daily'], df_coast['pred_idw_daily'])

    print(f"\n✅ 诊断结果：")
    print(f"【全局独立测试】纯 RF: {rmse_all_rf:.2f} -> RF-IDW: {rmse_all_idw:.2f}")
    print(f"【山区独立测试】纯 RF: {rmse_mt_rf:.2f} -> RF-IDW: {rmse_mt_idw:.2f}")
    print(f"【沿海独立测试】纯 RF: {rmse_coast_rf:.2f} -> RF-IDW: {rmse_coast_idw:.2f}")

    # ================= 绘图 =================
    labels = ['全局测试站点\n(All Test Sites)', '山区测试站点\n(DEM ≥ 50m)', '沿海测试站点\n(Lon ≥ 121.0°E)']
    rf_rmses = [rmse_all_rf, rmse_mt_rf, rmse_coast_rf]
    idw_rmses = [rmse_all_idw, rmse_mt_idw, rmse_coast_idw]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.bar(x - width/2, rf_rmses, width, label='原始随机森林模型 (RF)', color='#1f77b4', edgecolor='black', alpha=0.85)
    ax.bar(x + width/2, idw_rmses, width, label='混合残差校正模型 (RF-IDW)', color='#2ca02c', edgecolor='black', alpha=0.85)

    ax.set_ylabel(r'模型预测绝对误差 RMSE ($\mu g/m^3$)', fontproperties=my_font, fontsize=14)
    ax.set_title('空间独立框架下多区域残差校正验证', fontproperties=my_font, fontsize=16, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=my_font, fontsize=13)
    ax.legend(prop=my_font, fontsize=12, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, max(rf_rmses) * 1.25)

    for i in range(len(labels)):
        ax.annotate(f'{rf_rmses[i]:.2f}', xy=(x[i] - width/2, rf_rmses[i]), xytext=(0, 3), 
                    textcoords="offset points", ha='center', va='bottom', fontsize=12, fontfamily='serif', fontweight='bold')
        ax.annotate(f'{idw_rmses[i]:.2f}', xy=(x[i] + width/2, idw_rmses[i]), xytext=(0, 3), 
                    textcoords="offset points", ha='center', va='bottom', fontsize=12, fontfamily='serif', fontweight='bold')
        
        diff = rf_rmses[i] - idw_rmses[i]
        if diff > 0:
            ax.annotate(f'↓ {diff:.2f}', xy=(x[i] + width/2, idw_rmses[i] + 0.8), xytext=(0, 15), 
                        textcoords="offset points", ha='center', va='bottom', fontsize=12, color='red', fontweight='bold')
        elif diff < 0:
            ax.annotate(f'↑ {abs(diff):.2f}', xy=(x[i] + width/2, idw_rmses[i] + 0.8), xytext=(0, 15), 
                        textcoords="offset points", ha='center', va='bottom', fontsize=12, color='blue', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "Fig5_RF_IDW_Validation_Regions.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"\n🎉 完美出图！路径: {save_path}")

if __name__ == "__main__":
    main()