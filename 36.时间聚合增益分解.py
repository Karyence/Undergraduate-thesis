import os
import gc
import warnings
import pandas as pd
import numpy as np
import chinese_calendar as conc
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LogNorm 

try:
    import rmm
    rmm.reinitialize(managed_memory=True)
except ImportError:
    pass

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 路径与基础配置
# =============================================================================
BASE_DIR = "/home/wangzonghan/bisheshuju"
DATA_FILE = f"{BASE_DIR}/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"

HOURLY_MODEL_PATH = f"{BASE_DIR}/Results/Models_随机森林/best_rf_model_terrain.pkl"
HOURLY_FEAT_PATH = f"{BASE_DIR}/Results/Models_随机森林/best_features_list_terrain.pkl"

DAILY_MODEL_PATH = f"{BASE_DIR}/Results/Models_对照组_日均RF/best_rf_daily_model.pkl"
DAILY_FEAT_PATH = f"{BASE_DIR}/Results/Models_对照组_日均RF/best_features_list_daily.pkl"

# 字体配置
FONT_PATH = f"{BASE_DIR}/fonts/SimHei.ttf"
my_font = FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else FontProperties()
plt.rcParams['axes.unicode_minus'] = False

def calc_metrics(obs, pred):
    nmb = np.sum(pred - obs) / np.sum(obs) * 100
    r2 = r2_score(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    mae = mean_absolute_error(obs, pred)
    return r2, rmse, mae, nmb

def batch_predict(model, X, batch_size=20000):
    preds = []
    for i in range(0, X.shape[0], batch_size):
        batch_pred = model.predict(X[i : i + batch_size])
        if hasattr(batch_pred, 'to_numpy'): batch_pred = batch_pred.to_numpy()
        preds.extend(batch_pred)
    return np.array(preds)

# =============================================================================
# 🎨 核心出图模块：生成 SCI 出版级对比图表 
# =============================================================================
def generate_academic_figures(agg_df_daily, comp_df_hourly, output_dir):
    print("\n" + "🎨"*30)
    print("正在生成 SCI 出版级对比图表 ...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 图表 1：对数密度散点图
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)
    
    y_true = agg_df_daily['pm25_hourly'].values
    y_agg = agg_df_daily['pred_hourly'].values
    y_dir = agg_df_daily['Pred_Direct_Daily'].values
    
    max_val = max(np.max(y_true), np.max(y_agg), np.max(y_dir)) * 1.05
    
    def plot_density_scatter(ax, y_t, y_p, title):
        r2, rmse, mae, _ = calc_metrics(y_t, y_p)
        N = len(y_t)
        
        h = ax.hist2d(y_t, y_p, bins=100, cmap='jet', norm=LogNorm(), cmin=1)
        cb = fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('数据点密度 (个数)', fontproperties=my_font, fontsize=12)
        ax.plot([0, max_val], [0, max_val], 'k--', lw=2, label='1:1 Line')
        m, b = np.polyfit(y_t, y_p, 1)
        ax.plot(y_t, m*y_t + b, color='red', lw=2, label=f'Fit: y={m:.2f}x+{b:.2f}')
        
        textstr = f'N = {N:,}\n$R^2$ = {r2:.2f}\nRMSE = {rmse:.2f} $\mu g/m^3$\nMAE = {mae:.2f} $\mu g/m^3$'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props, family='serif')
        
        ax.set_title(title, fontproperties=my_font, fontsize=15, pad=15)
        ax.set_xlabel(r'真实 PM$_{2.5}$ 浓度 ($\mu g/m^3$)', fontproperties=my_font, fontsize=13)
        ax.set_ylabel(r'模型预测 PM$_{2.5}$ 浓度 ($\mu g/m^3$)', fontproperties=my_font, fontsize=13)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower right', frameon=True, edgecolor='black', prop=my_font, fontsize=10)

    plot_density_scatter(axes[0], y_true, y_agg, '独立测试集预测表现 (小时级训练+日均聚合模型)')
    plot_density_scatter(axes[1], y_true, y_dir, '独立测试集预测表现 (直接日均训练模型)')
    
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "Fig1_Scatter_Density_Comparison.png")
    plt.savefig(scatter_path, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 全中文对数密度散点图 已生成: {scatter_path}")

    # -------------------------------------------------------------------------
    # 图表 2：不同季节与污染等级下的 RMSE 精度提升双子图 
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # 跨坐标轴绘制置顶数值标签
    def add_top_level_labels(ax_base, ax_top, bars):
        for bar in bars:
            height = bar.get_height()
            ax_top.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 2),  # 向上偏移2个像素
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=11, fontfamily='serif',
                            xycoords=ax_base.transData,
                            zorder=20, # 强制置顶
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85)) 
    
    # === 子图 (a): 季节对比 ===
    season_names = ['冬季', '春季', '夏季', '秋季']
    rmse_agg_s, rmse_dir_s, rmse_diff_s = [], [], []
    for s_idx in [1, 2, 3, 4]:
        s_df = comp_df_hourly[comp_df_hourly['season'] == s_idx]
        _, r_a, _, _ = calc_metrics(s_df['pm25_hourly'].values, s_df['pred_hourly'].values)
        _, r_d, _, _ = calc_metrics(s_df['pm25_hourly'].values, s_df['Pred_Direct_Daily'].values)
        rmse_agg_s.append(r_a)
        rmse_dir_s.append(r_d)
        rmse_diff_s.append(r_d - r_a) 
        
    x = np.arange(len(season_names))
    width = 0.35
    
    ax1 = axes[0]
    bars1_agg = ax1.bar(x - width/2, rmse_agg_s, width, label='小时级训练+日均聚合', color='#1f77b4', edgecolor='black', alpha=0.85)
    bars1_dir = ax1.bar(x + width/2, rmse_dir_s, width, label='直接日均训练', color='#ff7f0e', edgecolor='black', alpha=0.85)
    ax1.set_ylabel(r'绝对均方根误差 RMSE ($\mu g/m^3$)', fontproperties=my_font, fontsize=14)
    ax1.set_title('(a) 不同季节下的高频追踪误差与精度提升', fontproperties=my_font, fontsize=16, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(season_names, fontproperties=my_font, fontsize=13)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, max(max(rmse_agg_s), max(rmse_dir_s)) * 1.2)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, rmse_diff_s, color='red', marker='D', markersize=8, linewidth=2.5, label='精度提升量 ($\Delta$RMSE)', zorder=10)
    ax1_twin.set_ylabel(r'精度提升量 $\Delta$RMSE ($\mu g/m^3$)', fontproperties=my_font, fontsize=14, color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1_twin.set_ylim(0, max(rmse_diff_s) * 1.5) 
    
    add_top_level_labels(ax1, ax1_twin, bars1_agg)
    add_top_level_labels(ax1, ax1_twin, bars1_dir)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', prop=my_font, fontsize=11, frameon=True)

    # === 子图 (b): 污染等级对比 ===
    bins = [0, 35, 75, 1000]
    labels = ['优良\n(<35)', '轻度污染\n(35-75)', '中重度污染\n(>=75)']
    comp_df_hourly['Level'] = pd.cut(comp_df_hourly['pm25_hourly'], bins=bins, labels=labels, right=False)
    
    rmse_agg_l, rmse_dir_l, rmse_diff_l = [], [], []
    for lvl in labels:
        l_df = comp_df_hourly[comp_df_hourly['Level'] == lvl]
        if len(l_df) > 0:
            _, r_a, _, _ = calc_metrics(l_df['pm25_hourly'].values, l_df['pred_hourly'].values)
            _, r_d, _, _ = calc_metrics(l_df['pm25_hourly'].values, l_df['Pred_Direct_Daily'].values)
            rmse_agg_l.append(r_a)
            rmse_dir_l.append(r_d)
            rmse_diff_l.append(r_d - r_a) 
            
    x_l = np.arange(len(labels))
    
    ax2 = axes[1]
    bars2_agg = ax2.bar(x_l - width/2, rmse_agg_l, width, label='小时级训练+日均聚合', color='#1f77b4', edgecolor='black', alpha=0.85)
    bars2_dir = ax2.bar(x_l + width/2, rmse_dir_l, width, label='直接日均训练', color='#ff7f0e', edgecolor='black', alpha=0.85) 
    ax2.set_ylabel(r'绝对均方根误差 RMSE ($\mu g/m^3$)', fontproperties=my_font, fontsize=14)
    ax2.set_title('(b) 不同污染等级下的高频追踪误差与精度提升', fontproperties=my_font, fontsize=16, pad=15)
    ax2.set_xticks(x_l)
    ax2.set_xticklabels(labels, fontproperties=my_font, fontsize=13)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, max(max(rmse_agg_l), max(rmse_dir_l)) * 1.2)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_l, rmse_diff_l, color='red', marker='D', markersize=8, linewidth=2.5, label='精度提升量 ($\Delta$RMSE)', zorder=10)
    ax2_twin.set_ylabel(r'精度提升量 $\Delta$RMSE ($\mu g/m^3$)', fontproperties=my_font, fontsize=14, color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2_twin.set_ylim(0, max(rmse_diff_l) * 1.3)
    
    add_top_level_labels(ax2, ax2_twin, bars2_agg)
    add_top_level_labels(ax2, ax2_twin, bars2_dir)
    
    for i, txt in enumerate(rmse_diff_l):
        ax2_twin.annotate(f'+{txt:.2f}', (x_l[i], rmse_diff_l[i]), textcoords="offset points", xytext=(0,10), ha='center', color='red', fontweight='bold', zorder=20)

    for i, txt in enumerate(rmse_diff_s):
        ax1_twin.annotate(f'+{txt:.2f}', (x[i], rmse_diff_s[i]), textcoords="offset points", xytext=(0,10), ha='center', color='red', fontweight='bold', zorder=20)

    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', prop=my_font, fontsize=11, frameon=True)

    plt.tight_layout()
    bar_path = os.path.join(output_dir, "Fig2_Performance_Improvement_BarChart.png")
    plt.savefig(bar_path, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 精度提升条形对比图 已生成: {bar_path}")
    print("🎨"*30 + "\n")

# =============================================================================
# 主程序入口
# =============================================================================
def main():
    print("="*85)
    print("🚀 启动：时间聚合增益四大维度全解构")
    print("="*85)
    
    # -------------------------------------------------------------------------
    # 1. 严格时间特征与数据加载
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 2. 推理预测 
    # -------------------------------------------------------------------------
    print("\n[2/4] 执行独立测试集推理...")
    hr_model = joblib.load(HOURLY_MODEL_PATH)
    test_df_hourly['pred_hourly'] = batch_predict(hr_model, test_df_hourly[joblib.load(HOURLY_FEAT_PATH)].astype('float32').values)
    del hr_model
    gc.collect()

    dy_model = joblib.load(DAILY_MODEL_PATH)
    test_df_daily['Pred_Direct_Daily'] = batch_predict(dy_model, test_df_daily[joblib.load(DAILY_FEAT_PATH)].astype('float32').values)
    del dy_model
    gc.collect()

    # -------------------------------------------------------------------------
    # 3. 数据融合
    # -------------------------------------------------------------------------
    print("\n[3/4] 正在构建微观小时级与宏观日均级比较...")
    
    comp_df_hourly = pd.merge(test_df_hourly, test_df_daily[['date', 'site_code', 'Pred_Direct_Daily']], on=['date', 'site_code'], how='inner')
    agg_df_daily = comp_df_hourly.groupby(['date', 'site_code'])[['pm25_hourly', 'pred_hourly', 'Pred_Direct_Daily']].mean().reset_index()

    # =========================================================================
    # 🎯 调用核心绘图模块，生成 SCI 级别可视化
    # =========================================================================
    OUTPUT_DIR = f"{BASE_DIR}/Results/Figures_时间聚合增益大考"
    generate_academic_figures(agg_df_daily, comp_df_hourly, OUTPUT_DIR)

if __name__ == "__main__":
    main()