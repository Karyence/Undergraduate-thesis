import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, train_test_split
from sklearn.cluster import KMeans  
from sklearn.inspection import permutation_importance
import warnings
import chinese_calendar as conc
import joblib
from cuml.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 基础配置
# =============================================================================
BASE_DIR = "/home/wangzonghan/bisheshuju"
DATA_FILE = f"{BASE_DIR}/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"
OUTPUT_DIR = f"{BASE_DIR}/Results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FONT_PATH = f"{BASE_DIR}/fonts/SimHei.ttf"
my_font = FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else FontProperties()
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 1. 数据加载与聚合 
# =============================================================================
def load_prepare_and_split():
    print("\n" + "="*70)
    print("📂 [1/5] 加载小时级数据集并构建对齐的日均特征池")
    print("="*70)
    
    df_hourly = pd.read_parquet(DATA_FILE)
    
    if 'month' not in df_hourly.columns:
        df_hourly['date'] = pd.to_datetime(df_hourly['date'])
        df_hourly['month'] = df_hourly['date'].dt.month
        df_hourly['day_of_week'] = df_hourly['date'].dt.dayofweek
        df_hourly['is_weekend'] = df_hourly['day_of_week'].isin([5, 6]).astype(int)
        df_hourly['season'] = (df_hourly['date'].dt.month % 12 // 3 + 1)
        df_hourly['is_holiday'] = df_hourly['date'].apply(lambda x: 1 if conc.is_holiday(x) else 0)
            
    sites_info = df_hourly[['site_code', 'lon', 'lat']].drop_duplicates().reset_index(drop=True)
    kmeans = KMeans(n_clusters=int(len(sites_info) * 0.15), random_state=42) 
    sites_info['spatial_cluster'] = kmeans.fit_predict(sites_info[['lon', 'lat']])
    test_sites_df = sites_info.groupby('spatial_cluster').apply(lambda x: x.sample(n=1, random_state=42)).reset_index(drop=True)
    
    test_sites = test_sites_df['site_code'].tolist()
    train_sites = [s for s in sites_info['site_code'] if s not in test_sites]

    print("  -> 正在将原生小时数据聚合成日均级矩阵...")
    numeric_cols = df_hourly.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['lon', 'lat', 'month', 'day_of_week', 'is_weekend', 'season', 'is_holiday', 'hour', 'date', 'datetime', 'site_code']
    cols_to_mean = [c for c in numeric_cols if c not in exclude_cols]
    
    df_daily = df_hourly.groupby(['date', 'site_code', 'lon', 'lat'])[cols_to_mean].mean().reset_index()
    
    df_daily['month'] = df_daily['date'].dt.month
    df_daily['day_of_week'] = df_daily['date'].dt.dayofweek
    df_daily['is_weekend'] = df_daily['day_of_week'].isin([5, 6]).astype(int)
    df_daily['season'] = (df_daily['month'] % 12 // 3 + 1).astype(int)
    df_daily['is_holiday'] = df_daily['date'].apply(lambda x: 1 if conc.is_holiday(x) else 0)

    train_df = df_daily[df_daily['site_code'].isin(train_sites)].copy()
    test_df = df_daily[df_daily['site_code'].isin(test_sites)].copy()
    
    exclude_cols_final = ['date', 'datetime', 'site_code', 'pm25_hourly', 'lon', 'lat']
    all_candidate_features = [c for c in df_daily.columns if c not in exclude_cols_final]
    
    print(f"  -> 测试样本: {len(test_df)} (目标: 14808 🎯)")
    return train_df, test_df, all_candidate_features

# =============================================================================
# 2. 特征筛选
# =============================================================================
def threshold_based_feature_selection(train_df, all_candidate_features):
    print("\n" + "="*70)
    print("🔍 [2/5] 执行全局重要性阈值特征筛选")
    print("="*70)
    
    sfs_sites = train_df['site_code'].unique()
    val_sites = pd.Series(sfs_sites).sample(frac=0.2, random_state=42).tolist()
    
    sfs_train_df = train_df[~train_df['site_code'].isin(val_sites)]
    sfs_val_df = train_df[train_df['site_code'].isin(val_sites)]
    
    y_train = sfs_train_df['pm25_hourly'].astype('float32').values
    y_val = sfs_val_df['pm25_hourly'].astype('float32').values
    X_train_full = sfs_train_df[all_candidate_features].astype('float32').values
    X_val_full = sfs_val_df[all_candidate_features].astype('float32').values
    
    eval_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    eval_model.fit(X_train_full, y_train)
    
    result = permutation_importance(eval_model, X_val_full, y_val, n_repeats=5, random_state=42, scoring='r2', n_jobs=1)
    
    threshold = 0.0001
    best_features = [f for i, f in enumerate(all_candidate_features) if result.importances_mean[i] > threshold]
    print(f"  🎯 筛选结束，保留特征数: {len(best_features)}")
    return best_features

# =============================================================================
# 3. 日尺度受限超参数寻优
# =============================================================================
def hyperparameter_tuning_and_evaluation(train_df, test_df, selected_features):
    print("\n" + "="*70)
    print("⚙️ [3/5] 执行日尺度空间寻优及评估")
    print("="*70)
    
    X_train = train_df[selected_features].astype('float32').values
    y_train = train_df['pm25_hourly'].astype('float32').values
    X_test = test_df[selected_features].astype('float32').values
    y_test = test_df['pm25_hourly'].astype('float32').values
    
    # 大幅削减深度，增加叶子节点样本限制，阻断模型过拟合
    param_dist = {
        'n_estimators': [100, 150], 
        'max_depth': [8, 10, 12, 14],            # 强制变浅
        'max_features': [0.3, 0.4, 0.5],         # 限制单树特征视野
        'min_samples_split': [4, 6],             # 增加分裂难度
        'min_samples_leaf': [2, 4],              # 增加叶子样本底线
        'bootstrap': [True]
    }
    
    rf = RandomForestRegressor(n_bins=128, random_state=42)
    groups = train_df['site_code'].values
    
    random_search = RandomizedSearchCV(
        estimator=rf, param_distributions=param_dist, n_iter=15, 
        cv=GroupKFold(n_splits=3), scoring='r2', n_jobs=1, verbose=0, random_state=42, error_score=0 
    )
    
    print("  ⏳ 启动范围随机寻优...")
    random_search.fit(X_train, y_train, groups=groups)
    best_model = random_search.best_estimator_
    print(f"\n  🏆 最终确定的正则化参数: {random_search.best_params_}")
    
    preds_daily = best_model.predict(X_test)
    if hasattr(preds_daily, 'to_numpy'): preds_daily = preds_daily.to_numpy()
        
    nmb = np.sum(preds_daily - y_test) / np.sum(y_test) * 100
    nme = np.sum(np.abs(preds_daily - y_test)) / np.sum(y_test) * 100
    r2_daily = r2_score(y_test, preds_daily)
    rmse_daily = np.sqrt(mean_squared_error(y_test, preds_daily))
    mae_daily = mean_absolute_error(y_test, preds_daily)
    
    print("\n" + "🔥"*30)
    print("  🎯 日均对照模型 终极评估报告 (Hold-out Sites)")
    print("🔥"*30)
    print(f"      -> 决定系数 R²   : {r2_daily:.4f}")
    print(f"      -> 均方根误差 RMSE: {rmse_daily:.2f} μg/m³")
    print(f"      -> 平均绝对误差 MAE: {mae_daily:.2f} μg/m³")
    print(f"      -> 标准化平均偏差 NMB: {nmb:+.2f}%")
    print("="*60)
    
    return best_model

# =============================================================================
# 4. 可视化与保存
# =============================================================================
def extract_and_plot_feature_importance(best_model, feature_names, test_df):
    X_test = test_df[feature_names].astype('float32').values
    y_test = test_df['pm25_hourly'].astype('float32').values 
    result = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=42, scoring='r2', n_jobs=1)
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': result.importances_mean}).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 10), dpi=300)
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='magma')
    plt.title('对照模型特征重要性评估 (Permutation - 日均模型)', fontproperties=my_font, fontsize=16)
    plt.tight_layout()
    
    fig_dir = os.path.join(OUTPUT_DIR, "Figures_对照组_日均RF")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "RF_Feature_Importance_Permutation_Daily.png"))
    plt.close()

if __name__ == "__main__":
    train_df, test_df, all_candidate_features = load_prepare_and_split()
    best_features = threshold_based_feature_selection(train_df, all_candidate_features)
    best_rf_model = hyperparameter_tuning_and_evaluation(train_df, test_df, best_features)
    extract_and_plot_feature_importance(best_rf_model, best_features, test_df)
    
    model_dir = os.path.join(OUTPUT_DIR, "Models_对照组_日均RF")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_rf_model, os.path.join(model_dir, "best_rf_daily_model.pkl"))
    joblib.dump(best_features, os.path.join(model_dir, "best_features_list_daily.pkl"))
    
    preds = best_rf_model.predict(test_df[best_features].astype('float32').values)
    if hasattr(preds, 'to_numpy'): preds = preds.to_numpy()
    test_df['predicted_pm25'] = preds
    test_df.to_csv(os.path.join(model_dir, "Daily_Model_Test_Results_14808.csv"), index=False)
    print("🎉 文件落盘全部完成！")