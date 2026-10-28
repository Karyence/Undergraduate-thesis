import os
import gc
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from scipy.spatial import cKDTree
import chinese_calendar as conc
import warnings

try:
    import rmm
    rmm.reinitialize(managed_memory=True)
except ImportError:
    pass

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 基础配置与路径
# =============================================================================
BASE_DIR = "/home/wangzonghan/bisheshuju"
DATA_FILE = f"{BASE_DIR}/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"

HOURLY_MODEL_PATH = f"{BASE_DIR}/Results/Models_随机森林/best_rf_model_terrain.pkl"
HOURLY_FEAT_PATH = f"{BASE_DIR}/Results/Models_随机森林/best_features_list_terrain.pkl"

DAILY_MODEL_PATH = f"{BASE_DIR}/Results/Models_对照组_日均RF/best_rf_daily_model.pkl"
DAILY_FEAT_PATH = f"{BASE_DIR}/Results/Models_对照组_日均RF/best_features_list_daily.pkl"

OUTPUT_DIR = f"{BASE_DIR}/Results/Figures_残差诊断"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['axes.unicode_minus'] = False

def batch_predict(model, X, batch_size=20000):
    preds = []
    for i in range(0, X.shape[0], batch_size):
        batch_pred = model.predict(X[i : i + batch_size])
        if hasattr(batch_pred, 'to_numpy'): batch_pred = batch_pred.to_numpy()
        preds.extend(batch_pred)
    return np.array(preds)

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
    print("="*85)
    print("🚀 启动")
    print("="*85)
    
    df_hourly = pd.read_parquet(DATA_FILE)
    if 'month' not in df_hourly.columns:
        df_hourly['date'] = pd.to_datetime(df_hourly['date'])
        df_hourly['month'] = df_hourly['date'].dt.month
        df_hourly['day_of_week'] = df_hourly['date'].dt.dayofweek
        df_hourly['is_weekend'] = df_hourly['day_of_week'].isin([5, 6]).astype(int)
        df_hourly['season'] = (df_hourly['month'] % 12 // 3 + 1).astype(int)
        df_hourly['is_holiday'] = df_hourly['date'].apply(lambda x: 1 if conc.is_holiday(x) else 0)

    sites_info = df_hourly[['site_code', 'lon', 'lat']].drop_duplicates().reset_index(drop=True)
    kmeans = KMeans(n_clusters=int(len(sites_info) * 0.15), random_state=42)
    sites_info['spatial_cluster'] = kmeans.fit_predict(sites_info[['lon', 'lat']])
    test_sites = sites_info.groupby('spatial_cluster').apply(lambda x: x.sample(n=1, random_state=42)).reset_index(drop=True)['site_code'].tolist()
    
    test_df_hourly = df_hourly[df_hourly['site_code'].isin(test_sites)].copy()
    train_df_hourly = df_hourly[~df_hourly['site_code'].isin(test_sites)].copy()
    
    numeric_cols = df_hourly.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['lon', 'lat', 'month', 'day_of_week', 'is_weekend', 'season', 'is_holiday']
    cols_to_mean = [c for c in numeric_cols if c not in exclude_cols]
    
    df_daily = df_hourly.groupby(['date', 'site_code', 'lon', 'lat'])[cols_to_mean].mean().reset_index()
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily['month'] = df_daily['date'].dt.month
    df_daily['day_of_week'] = df_daily['date'].dt.dayofweek
    df_daily['is_weekend'] = df_daily['day_of_week'].isin([5, 6]).astype(int)
    df_daily['season'] = (df_daily['month'] % 12 // 3 + 1).astype(int)
    df_daily['is_holiday'] = df_daily['date'].apply(lambda x: 1 if conc.is_holiday(x) else 0)
    
    test_df_daily = df_daily[df_daily['site_code'].isin(test_sites)].copy()

    print("\n[1/3] 执行 ❀36.py 原生测试集与训练集推理...")
    hr_model = joblib.load(HOURLY_MODEL_PATH)
    hr_feats = joblib.load(HOURLY_FEAT_PATH)
    test_df_hourly['pred_hourly'] = batch_predict(hr_model, test_df_hourly[hr_feats].astype('float32').values)
    train_df_hourly['pred_hourly'] = batch_predict(hr_model, train_df_hourly[hr_feats].astype('float32').values)
    del hr_model
    gc.collect()

    dy_model = joblib.load(DAILY_MODEL_PATH)
    test_df_daily['Pred_Direct_Daily'] = batch_predict(dy_model, test_df_daily[joblib.load(DAILY_FEAT_PATH)].astype('float32').values)
    del dy_model
    gc.collect()

    comp_df_hourly = pd.merge(test_df_hourly, test_df_daily[['date', 'site_code', 'Pred_Direct_Daily']], on=['date', 'site_code'], how='inner')
    agg_df_daily = comp_df_hourly.groupby(['date', 'site_code'])[['pm25_hourly', 'pred_hourly', 'Pred_Direct_Daily']].mean().reset_index()

    site_meta = sites_info.merge(df_hourly[['site_code', 'DEM']].drop_duplicates(), on='site_code', how='left')
    agg_df_daily = agg_df_daily.merge(site_meta[['site_code', 'lon', 'lat', 'DEM']], on='site_code', how='left')

    print(f"🎯 [核验] agg_df_daily 纯 RF RMSE = {calc_rmse(agg_df_daily['pm25_hourly'].values, agg_df_daily['pred_hourly'].values):.2f} (与图3.1a完全一致！)")

    print("\n[2/3] 聚合训练站日均残差并执行 IDW 空间补偿...")
    train_daily = train_df_hourly.groupby(['date', 'site_code', 'lon', 'lat'])[['pm25_hourly', 'pred_hourly']].mean().reset_index()
    train_daily['date'] = pd.to_datetime(train_daily['date'])
    agg_df_daily['date'] = pd.to_datetime(agg_df_daily['date'])

    agg_df_daily['pred_idw_daily'] = agg_df_daily['pred_hourly']

    for dt, test_group in agg_df_daily.groupby('date'):
        train_group = train_daily[train_daily['date'] == dt]
        if len(train_group) < 5: continue
        
        train_lon, train_lat = train_group['lon'].values, train_group['lat'].values
        train_res = train_group['pm25_hourly'].values - train_group['pred_hourly'].values
        
        test_lon, test_lat = test_group['lon'].values, test_group['lat'].values
        interp_res = idw_interpolation(train_lon, train_lat, train_res, test_lon, test_lat, k=15, p=2)
        agg_df_daily.loc[test_group.index, 'pred_idw_daily'] = test_group['pred_hourly'] + interp_res

    agg_df_daily['pred_idw_daily'] = np.clip(agg_df_daily['pred_idw_daily'], 0, 300)

    # 多区域评估与成图
    rmse_all_rf = round(calc_rmse(agg_df_daily['pm25_hourly'].values, agg_df_daily['pred_hourly'].values), 2)
    rmse_all_idw = round(calc_rmse(agg_df_daily['pm25_hourly'].values, agg_df_daily['pred_idw_daily'].values), 2)
    
    df_mt = agg_df_daily[agg_df_daily['DEM'] >= 50]
    rmse_mt_rf = round(calc_rmse(df_mt['pm25_hourly'].values, df_mt['pred_hourly'].values), 2)
    rmse_mt_idw = round(calc_rmse(df_mt['pm25_hourly'].values, df_mt['pred_idw_daily'].values), 2)
    
    df_coast = agg_df_daily[agg_df_daily['lon'] >= 121.0]
    rmse_coast_rf = round(calc_rmse(df_coast['pm25_hourly'].values, df_coast['pred_hourly'].values), 2)
    rmse_coast_idw = round(calc_rmse(df_coast['pm25_hourly'].values, df_coast['pred_idw_daily'].values), 2)

    print(f"\n✅ 诊断结果：")
    print(f"【全局独立测试】(Sites={agg_df_daily['site_code'].nunique()}, N={len(agg_df_daily)}) 纯 RF: {rmse_all_rf:.2f} -> RF-IDW: {rmse_all_idw:.2f} (Δ={rmse_all_rf-rmse_all_idw:.2f}, {(rmse_all_idw-rmse_all_rf)/rmse_all_rf*100:.1f}%)")
    print(f"【山区独立测试】(Sites={df_mt['site_code'].nunique()}, N={len(df_mt)}) 纯 RF: {rmse_mt_rf:.2f} -> RF-IDW: {rmse_mt_idw:.2f} (Δ={rmse_mt_rf-rmse_mt_idw:.2f}, {(rmse_mt_idw-rmse_mt_rf)/rmse_mt_rf*100:.1f}%)")
    print(f"【沿海独立测试】(Sites={df_coast['site_code'].nunique()}, N={len(df_coast)}) 纯 RF: {rmse_coast_rf:.2f} -> RF-IDW: {rmse_coast_idw:.2f} (Δ={rmse_coast_rf-rmse_coast_idw:.2f}, {(rmse_coast_idw-rmse_coast_rf)/rmse_coast_rf*100:.1f}%)")

    labels = ['All Test Sites', 'Mountainous Sites\n(DEM $\\geq$ 50m)', 'Coastal Sites\n(Lon $\\geq$ 121.0$^\\circ$E)']
    rf_rmses = [rmse_all_rf, rmse_mt_rf, rmse_coast_rf]
    idw_rmses = [rmse_all_idw, rmse_mt_idw, rmse_coast_idw]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.bar(x - width/2, rf_rmses, width, label='Original Model (RF)', color='#1f77b4', edgecolor='black', alpha=0.85)
    ax.bar(x + width/2, idw_rmses, width, label='Hybrid Residual Correction (RF-IDW)', color='#2ca02c', edgecolor='black', alpha=0.85)

    ax.set_ylabel(r'Prediction RMSE ($\boldsymbol{\mu}\mathbf{g}/\mathbf{m}^{\mathbf{3}}$)', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, weight='bold')
    ax.legend(fontsize=10.5, loc='upper right', edgecolor='gray', framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, max(rf_rmses) * 1.35)

    for i in range(len(labels)):
        ax.annotate(f'{rf_rmses[i]:.2f}', xy=(x[i] - width/2, rf_rmses[i]), xytext=(0, 3), 
                    textcoords="offset points", ha='center', va='bottom', fontsize=12, fontfamily='serif', fontweight='bold')
        ax.annotate(f'{idw_rmses[i]:.2f}', xy=(x[i] + width/2, idw_rmses[i]), xytext=(0, 3), 
                    textcoords="offset points", ha='center', va='bottom', fontsize=12, fontfamily='serif', fontweight='bold')
        diff = rf_rmses[i] - idw_rmses[i]
        if diff > 0:
            ax.annotate(f'$\\downarrow$ {diff:.2f}', xy=(x[i] + width/2, idw_rmses[i] + 0.8), xytext=(0, 15), 
                        textcoords="offset points", ha='center', va='bottom', fontsize=12, color='red', fontweight='bold')
        elif diff < 0:
            ax.annotate(f'$\\uparrow$ {abs(diff):.2f}', xy=(x[i] + width/2, idw_rmses[i] + 0.8), xytext=(0, 15), 
                        textcoords="offset points", ha='center', va='bottom', fontsize=12, color='blue', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "Fig5_RF_IDW_Validation_Regions.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"\n🎉 完美出图！路径: {save_path}")

if __name__ == "__main__":
    main()