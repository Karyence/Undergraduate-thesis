import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import joblib
import warnings
import chinese_calendar as conc 

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 基础路径配置
# =============================================================================
DATA_FILE = "/home/wangzonghan/bisheshuju/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"
OUTPUT_DIR = "/home/wangzonghan/bisheshuju/Results/"

# 加载刚训练好的模型和特征列表
MODEL_PATH = os.path.join(OUTPUT_DIR, "Models_随机森林", "best_rf_model_terrain.pkl")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "Models_随机森林", "best_features_list_terrain.pkl")

FONT_PATH = "/home/wangzonghan/bisheshuju/fonts/SimHei.ttf"
my_font = FontProperties(fname=FONT_PATH, size=14) if os.path.exists(FONT_PATH) else FontProperties(size=14)
plt.rcParams['axes.unicode_minus'] = False

def get_test_data():
    print("\n" + "="*70)
    print("📂 [1/3] 加载数据与复原测试集 (Hold-out Sites)")
    print("="*70)
    
    df = pd.read_parquet(DATA_FILE)
    
    # 动态生成时间特征，补齐 23 个特征的闭环！
    if 'month' not in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['season'] = (df['date'].dt.month % 12 // 3 + 1)
        df['is_holiday'] = df['date'].apply(lambda x: 1 if conc.is_holiday(x) else 0)
    
    sites_info = df[['site_code', 'lon', 'lat']].drop_duplicates().reset_index(drop=True)
    num_test = int(len(sites_info) * 0.15)
    
    kmeans = KMeans(n_clusters=num_test, random_state=42)
    sites_info['spatial_cluster'] = kmeans.fit_predict(sites_info[['lon', 'lat']])
    
    test_sites_df = sites_info.groupby('spatial_cluster').apply(
        lambda x: x.sample(n=1, random_state=42)
    ).reset_index(drop=True)
    
    test_sites = test_sites_df['site_code'].tolist()
    test_df = df[df['site_code'].isin(test_sites)].copy()
    
    print(f"  ✅ 成功还原独立测试集: 包含 {len(test_sites)} 个站点, 共 {len(test_df):,} 行小时样本。")
    return test_df

def make_predictions(test_df):
    print("\n" + "="*70)
    print("⚙️ [2/3] 加载 RF 模型并执行双尺度预测")
    print("="*70)
    
    best_model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    
    X_test = test_df[features].astype('float32').values
    
    print("  ⏳ 正在进行小时级高频预测 (二十多万行，请稍候)...")
    preds_hourly = best_model.predict(X_test)
    if hasattr(preds_hourly, 'to_numpy'): preds_hourly = preds_hourly.to_numpy()
    
    # 构造评估 DataFrame
    eval_df = test_df[['date', 'site_code', 'pm25_hourly']].copy()
    eval_df['pred_hourly'] = preds_hourly
    
    # 聚合得到日均尺度
    daily_eval_df = eval_df.groupby(['date', 'site_code']).mean().reset_index()
    
    return eval_df, daily_eval_df

def plot_dual_scale_scatter(hourly_df, daily_df):
    print("\n" + "="*70)
    print("🎨 [3/3] 绘制并保存双尺度密度散点图")
    print("="*70)
    
    # 稍微调整画布整体比例，适应两个正方形的子图
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=300)
    
    def draw_scatter(ax, y_true, y_pred, title, is_hourly=True):
        # 计算评价指标
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        n_samples = len(y_true)
        
        limit_percentile = 99.8
        max_val = max(np.percentile(y_true, limit_percentile), np.percentile(y_pred, limit_percentile))
        # 向上取整到 50 的倍数
        max_val = np.ceil(max_val / 50) * 50 
        
        # 强制锁定 1:1 物理正方形比例
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        
        bins = 150 if is_hourly else 80
        h = ax.hist2d(y_true, y_pred, bins=bins, range=[[0, max_val], [0, max_val]], 
                      cmap='jet', cmin=1, norm=LogNorm())
        
        # 添加 1:1 对角线和回归拟合线
        ax.plot([0, max_val], [0, max_val], 'k--', lw=2, label='1:1 Line')
        m, b = np.polyfit(y_true, y_pred, 1)
        ax.plot(y_true, m*y_true + b, color='red', lw=2, label=f'Fit: y={m:.2f}x+{b:.2f}')
        
        textstr = '\n'.join((
            f'N = {n_samples:,}',
            f'$R^2$ = {r2:.2f}', 
            f'RMSE = {rmse:.2f} $\\mu g/m^3$',
            f'MAE = {mae:.2f} $\\mu g/m^3$'
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props, fontproperties=my_font)
        
        # 设置标签和标题
        ax.set_xlabel('真实 PM$_{2.5}$ 浓度 ($\\mu g/m^3$)', fontproperties=my_font)
        ax.set_ylabel('模型预测 PM$_{2.5}$ 浓度 ($\\mu g/m^3$)', fontproperties=my_font)
        ax.set_title(title, fontproperties=my_font, fontsize=16, pad=15)
        ax.legend(loc='lower right', prop={'size': 11})
        ax.grid(True, linestyle=':', alpha=0.6)
        
        return h[3] # 返回 colormap object 用于画 colorbar
    # 1. 绘制左图：小时级
    im1 = draw_scatter(axes[0], hourly_df['pm25_hourly'].values, hourly_df['pred_hourly'].values, 
                       '独立测试集预测表现 (小时尺度 Hourly)', is_hourly=True)
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('数据点密度 (Count)', fontproperties=my_font)
    
    # 2. 绘制右图：日均级
    im2 = draw_scatter(axes[1], daily_df['pm25_hourly'].values, daily_df['pred_hourly'].values, 
                       '独立测试集预测表现 (日均尺度 Daily)', is_hourly=False)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('数据点密度 (Count)', fontproperties=my_font)
    
    plt.tight_layout()
    
    # 保存结果
    save_dir = os.path.join(OUTPUT_DIR, "Figures_随机森林")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "Dual_Scale_Scatter_Density.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"  🎉 大功告成！完美 1:1 比例对比散点图已保存至:\n  {save_path}")

if __name__ == "__main__":
    test_df = get_test_data()
    hourly_df, daily_df = make_predictions(test_df)
    plot_dual_scale_scatter(hourly_df, daily_df)