import os
import gc
import warnings
import pandas as pd
import numpy as np
import chinese_calendar as conc
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 路径与基础配置
# =============================================================================
BASE_DIR = "/home/wangzonghan/bisheshuju"
DATA_FILE = f"{BASE_DIR}/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"
OUTPUT_DIR = f"{BASE_DIR}/Results/Figures_消融实验"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HOURLY_FEAT_PATH = f"{BASE_DIR}/Results/Models_随机森林/best_features_list_terrain.pkl"

FONT_PATH = f"{BASE_DIR}/fonts/SimHei.ttf"
my_font = FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else FontProperties()
plt.rcParams['axes.unicode_minus'] = False

def calc_metrics(obs, pred):
    r2 = r2_score(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    return r2, rmse

# =============================================================================
# 1. 严格空间隔离与数据加载
# =============================================================================
def load_and_split_data():
    print("="*80)
    print("📂 [1/3] 加载全量数据并执行严格 K-Means 空间隔离...")
    df = pd.read_parquet(DATA_FILE)
    
    if 'month' not in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['season'] = (df['month'] % 12 // 3 + 1).astype(int)
        df['is_holiday'] = df['date'].apply(lambda x: 1 if conc.is_holiday(x) else 0)

    sites_info = df[['site_code', 'lon', 'lat']].drop_duplicates().reset_index(drop=True)
    kmeans = KMeans(n_clusters=int(len(sites_info) * 0.15), random_state=42)
    sites_info['spatial_cluster'] = kmeans.fit_predict(sites_info[['lon', 'lat']])
    test_sites = sites_info.groupby('spatial_cluster').apply(lambda x: x.sample(n=1, random_state=42)).reset_index(drop=True)['site_code'].tolist()
    
    train_df = df[~df['site_code'].isin(test_sites)].copy()
    test_df = df[df['site_code'].isin(test_sites)].copy()
    
    best_features = joblib.load(HOURLY_FEAT_PATH)
    print(f"  ✅ 训练集样本数: {len(train_df):,} | 测试集样本数: {len(test_df):,} | 基础特征数: {len(best_features)}")
    return train_df, test_df, best_features

# =============================================================================
# 2. 消融实验平行训练引擎 
# =============================================================================
def run_ablation_experiments(train_df, test_df, base_features):
    print("\n" + "="*80)
    print("🚀 [2/3] 启动核心消融实验 ...")
    
    experiments = {
        'Baseline': {'desc': '完整多频率特征', 'drop': []},
        'No_AOD': {'desc': '去除卫星 AOD', 'drop': ['AOD']},
        'No_Hourly_Met': {'desc': '去除小时气象动力场', 'drop': ['ERA5_BLH', 'ERA5_T2M', 'ERA5_D2M', 'ERA5_WIND', 'ERA5_TP']}, 
        'No_Static': {'desc': '去除静态地理信息', 'drop': ['DEM', 'Slope', 'POP'] + [f for f in base_features if f.startswith('LC_')]},
        'No_Time': {'desc': '去除动态时间周期', 'drop': ['hour', 'month', 'season', 'day_of_week', 'is_weekend', 'is_holiday']}
    }

    results = []
    y_train = train_df['pm25_hourly'].values
    y_test = test_df['pm25_hourly'].values

    for exp_name, setup in experiments.items():
        print(f"\n  ▶ 正在执行 [{setup['desc']}]...")
        current_features = [f for f in base_features if f not in setup['drop']]
        print(f"    (当前入模特征数量: {len(current_features)})")
        
        X_tr = train_df[current_features].copy()
        X_te = test_df[current_features].copy()

        # 保持公平的最优主模型参数
        model = RandomForestRegressor(n_estimators=200, max_depth=22, max_features=0.6, 
                                      min_samples_split=2, random_state=42, n_jobs=-1)
        model.fit(X_tr.values, y_train)
        preds = model.predict(X_te.values)
        
        r2, rmse = calc_metrics(y_test, preds)
        print(f"    ✅ R²: {r2:.4f} | RMSE: {rmse:.2f} μg/m³")
        
        results.append({
            'Experiment': setup['desc'],
            'RMSE': rmse,
            'R2': r2
        })
        
        del model, X_tr, X_te
        gc.collect()

    return pd.DataFrame(results)

# =============================================================================
# 3. 生成 SCI 出版级消融对比图
# =============================================================================
def plot_ablation_results(res_df):
    print("\n" + "="*80)
    print("🎨 [3/3] 正在生成消融实验对比图...")
    
    fig, ax1 = plt.subplots(figsize=(13, 7), dpi=300)
    
    x = np.arange(len(res_df))
    width = 0.5
    
    base_rmse = res_df.iloc[0]['RMSE']
    res_df['RMSE_Increase'] = res_df['RMSE'] - base_rmse

    colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd']
    bars = ax1.bar(x, res_df['RMSE'], width, color=colors, edgecolor='black', alpha=0.85)
    
    ax1.set_ylabel(r'模型预测绝对误差 RMSE ($\mu g/m^3$)', fontproperties=my_font, fontsize=15)
    ax1.set_xticks(x)
    # 标签换行，防止挤压
    labels = [desc.replace(' ', '\n') for desc in res_df['Experiment']]
    ax1.set_xticklabels(labels, fontproperties=my_font, fontsize=13)
    ax1.set_ylim(0, max(res_df['RMSE']) * 1.25)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=12, fontweight='bold', fontfamily='serif')
        if i > 0:
            inc = res_df.iloc[i]['RMSE_Increase']
            ax1.annotate(f'+{inc:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height + 0.8),
                        xytext=(0, 15), textcoords="offset points", ha='center', va='bottom',
                        fontsize=12, color='red', fontweight='bold', fontfamily='serif')

    ax2 = ax1.twinx()
    ax2.plot(x, res_df['R2'], color='black', marker='o', markersize=8, linewidth=2.5, linestyle='--', label='决定系数 ($R^2$)')
    ax2.set_ylabel(r'模型拟合度决定系数 ($R^2$)', fontproperties=my_font, fontsize=15)
    # 动态调整右轴下界，让折线不要贴底
    ax2.set_ylim(min(res_df['R2']) - 0.05, 1.0)
    
    for i, r2_val in enumerate(res_df['R2']):
        ax2.annotate(f'{r2_val:.3f}', xy=(x[i], r2_val), xytext=(0, 10), textcoords="offset points",
                     ha='center', va='bottom', fontsize=12, fontfamily='serif')

    ax1.set_title('长三角多频率特征融合消融实验综合评估 (Ablation Study)', fontproperties=my_font, fontsize=17, pad=20)
    
    # 合并双轴图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', prop=my_font, fontsize=12, frameon=True)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "Fig3_Ablation_Study_BarChart.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 完美！全中文消融实验条形图已保存至: {save_path}")
    print("="*80)

if __name__ == "__main__":
    df_tr, df_te, b_feats = load_and_split_data()
    res = run_ablation_experiments(df_tr, df_te, b_feats)
    plot_ablation_results(res)