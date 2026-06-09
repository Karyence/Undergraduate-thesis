import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import chinese_calendar as conc
import joblib
import warnings

from catboost import CatBoostRegressor 

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 基础路径配置
# =============================================================================
BASE_DIR = "/home/wangzonghan/bisheshuju"
DATA_FILE = f"{BASE_DIR}/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"

RF_MODEL_DIR = f"{BASE_DIR}/Results/Models_随机森林"
CB_MODEL_DIR = f"{BASE_DIR}/Results/Models_CatBoost"
COMPARE_FIG_DIR = f"{BASE_DIR}/Results/Figures_模型对比"

os.makedirs(COMPARE_FIG_DIR, exist_ok=True)

# 字体配置 
FONT_PATH = f"{BASE_DIR}/fonts/SimHei.ttf"
TITLE_FS = 20    # 标题字号
LABEL_FS = 16    # 轴标签字号
TICK_FS = 14     # 刻度字号
TEXT_FS = 15     # 统计框字号
LEGEND_FS = 14   # 图例字号

if os.path.exists(FONT_PATH):
    my_font = FontProperties(fname=FONT_PATH)
else:
    my_font = FontProperties()
    
plt.rcParams['axes.unicode_minus'] = False

def calculate_metrics(obs, pred):
    """计算用于表格输出的核心统计指标"""
    r2 = r2_score(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    mae = mean_absolute_error(obs, pred)
    nmb = np.sum(pred - obs) / np.sum(obs) * 100
    return r2, rmse, mae, nmb

def main():
    print("="*70)
    print("🚀 启动：30.双模型小时级性能对比 (RF vs CatBoost)")
    print("="*70)

    # =============================================================================
    # 1. 加载数据与构建统一独立测试集 
    # =============================================================================
    print("\n📂 [1/3] 加载数据并复刻 15% 空间独立测试集...")
    df = pd.read_parquet(DATA_FILE)
    
    # 补全时间特征
    if 'month' not in df.columns or 'season' not in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['season'] = (df['date'].dt.month % 12 // 3 + 1)
        df['is_holiday'] = df['date'].apply(lambda x: 1 if conc.is_holiday(x) else 0)

    best_features = joblib.load(os.path.join(RF_MODEL_DIR, "best_features_list_terrain.pkl"))

    # K-Means 空间隔离
    sites_info = df[['site_code', 'lon', 'lat']].drop_duplicates().reset_index(drop=True)
    num_test = int(len(sites_info) * 0.15)
    kmeans = KMeans(n_clusters=num_test, random_state=42)
    sites_info['spatial_cluster'] = kmeans.fit_predict(sites_info[['lon', 'lat']])
    
    test_sites_df = sites_info.groupby('spatial_cluster').apply(lambda x: x.sample(n=1, random_state=42)).reset_index(drop=True)
    test_sites = test_sites_df['site_code'].tolist()
    
    test_df = df[df['site_code'].isin(test_sites)].copy()
    
    X_test = test_df[best_features].astype('float32').values
    y_test_hourly = test_df['pm25_hourly'].astype('float32').values

    # =============================================================================
    # 2. 加载模型执行小时级预测
    # =============================================================================
    print("\n⚙️ [2/3] 加载 RF 和 CatBoost 模型执行小时级盲测...")
    rf_model = joblib.load(os.path.join(RF_MODEL_DIR, "best_rf_model_terrain.pkl"))
    cb_model = joblib.load(os.path.join(CB_MODEL_DIR, "best_cb_model_terrain.pkl"))

    preds_rf_hourly = rf_model.predict(X_test)
    preds_cb_hourly = cb_model.predict(X_test)

    # =============================================================================
    # 3. 绘制 1x2 左右对比图
    # =============================================================================
    print("\n🎨 [3/3] 正在渲染 1x2 极致清晰对比散点图...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), dpi=300)
    
    def draw_single_scatter(ax, y_true, y_pred, title):
        n_samples = len(y_true)
        r2, rmse, mae, nmb = calculate_metrics(y_true, y_pred)
        
        # 统一坐标轴范围
        max_val = max(np.percentile(y_true, 99.9), np.percentile(y_pred, 99.9))
        max_val = np.ceil(max_val / 20) * 20 
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        
        # 绘制密度散点
        h = ax.hist2d(y_true, y_pred, bins=150, range=[[0, max_val], [0, max_val]], 
                      cmap='jet', cmin=1, norm=LogNorm())
        
        # 1:1 线与回归拟合线 
        ax.plot([0, max_val], [0, max_val], 'k--', lw=2.5, label='1:1 Line', alpha=0.8)
        m, b = np.polyfit(y_true, y_pred, 1)
        ax.plot(y_true, m*y_true + b, color='red', lw=3.0, label=f'Fit: y={m:.2f}x+{b:.2f}')
        
        # 统计信息框 
        textstr = '\n'.join((
            f'N = {n_samples:,}',
            f'$R^2$ = {r2:.4f}', 
            f'RMSE = {rmse:.2f} $\\mu g/m^3$',
            f'NMB = {nmb:+.2f}%'
        ))
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.85, edgecolor='gray')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=TEXT_FS, fontproperties=my_font,
                verticalalignment='top', bbox=props)
        
        # 标题与标签
        ax.set_title(title, fontproperties=my_font, fontsize=TITLE_FS, weight='bold', pad=20)
        ax.set_xlabel('真实观测 PM$_{2.5}$ 浓度 ($\\mu g/m^3$)', fontproperties=my_font, fontsize=LABEL_FS)
        ax.set_ylabel('模型预测 PM$_{2.5}$ 浓度 ($\\mu g/m^3$)', fontproperties=my_font, fontsize=LABEL_FS)
        
        # 刻度字号
        ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
        ax.legend(loc='lower right', prop={'family': my_font.get_name(), 'size': LEGEND_FS})
        ax.grid(True, linestyle=':', alpha=0.5)
        return h[3]

    # 绘制左右两图
    im1 = draw_single_scatter(axes[0], y_test_hourly, preds_rf_hourly, "(a) 随机森林 (Random Forest)")
    im2 = draw_single_scatter(axes[1], y_test_hourly, preds_cb_hourly, "(b) CatBoost")

    # 调整布局
    plt.tight_layout(rect=[0, 0, 0.93, 1]) 
    
    # 添加垂直色标
    cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('小时样本点密度 (Count)', fontproperties=my_font, fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)

    # 保存文件
    save_path = os.path.join(COMPARE_FIG_DIR, "Models_Comparison_Hourly_Scatter.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"✅ 1x2 小时级对比图已生成！\n 📂 路径: {save_path}")
    print(f"\n📊 快速对照指标:")
    print(f"   - RF: R²={r2_score(y_test_hourly, preds_rf_hourly):.4f}")
    print(f"   - CB: R²={r2_score(y_test_hourly, preds_cb_hourly):.4f}")

if __name__ == "__main__":
    main()