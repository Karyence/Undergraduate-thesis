import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.cluster import KMeans
from sklearn.inspection import partial_dependence
from scipy.ndimage import gaussian_filter1d
import chinese_calendar as conc
import joblib
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 基础路径配置
# =============================================================================
BASE_DIR = "/home/wangzonghan/bisheshuju"
DATA_FILE = f"{BASE_DIR}/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"
RF_MODEL_DIR = f"{BASE_DIR}/Results/Models_随机森林"
FIG_DIR = f"{BASE_DIR}/Results/Figures_随机森林"

os.makedirs(FIG_DIR, exist_ok=True)

FONT_PATH = f"{BASE_DIR}/fonts/SimHei.ttf"
my_font = FontProperties(fname=FONT_PATH, size=14) if os.path.exists(FONT_PATH) else FontProperties(size=14)
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("="*70)
    print("🚀 启动：31. 模型可解释性分析 ")
    print("="*70)

    # =============================================================================
    # 1. 严格复刻独立测试集
    # =============================================================================
    print("\n📂 [1/3] 加载数据与复原测试集...")
    df = pd.read_parquet(DATA_FILE)
    
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
    
    test_sites_df = sites_info.groupby('spatial_cluster').apply(lambda x: x.sample(n=1, random_state=42)).reset_index(drop=True)
    test_df = df[df['site_code'].isin(test_sites_df['site_code'].tolist())].copy()

    # =============================================================================
    # 2. 加载模型与特征
    # =============================================================================
    print(" ⏳ [2/3] 加载 Random Forest 最优模型...")
    rf_model = joblib.load(os.path.join(RF_MODEL_DIR, "best_rf_model_terrain.pkl"))
    best_features = joblib.load(os.path.join(RF_MODEL_DIR, "best_features_list_terrain.pkl"))
    
    X_test = test_df[best_features].astype('float32')

    # 核心物理特征列表及其在图表中的展示名称
    feature_dict = {
        'AOD': '气溶胶光学厚度 (AOD)',
        'ERA5_BLH': '边界层高度 (BLH, m)',
        'ERA5_T2M': '2米地表温度 (T2M, ℃)',
        'ERA5_WIND': '地表风速 (Wind, m/s)',
        'ERA5_D2M': '2米露点温度 (D2M, ℃)',
        'DEM': '陆地海拔高度 (DEM, m)'
    }
    valid_features = [f for f in feature_dict.keys() if f in best_features]

    # =============================================================================
    # 3. 手工计算并绘制 PDP 曲线
    # =============================================================================
    print("\n🎨 [3/3] 正在提取非线性响应数据并渲染精美图表...")
    
    X_sample = X_test.sample(n=min(30000, len(X_test)), random_state=42)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes = axes.flatten()

    for idx, feature in enumerate(valid_features):
        ax = axes[idx]
        print(f"   -> 正在计算 {feature} 的响应曲线...")

        pdp_results = partial_dependence(rf_model, X_sample, [feature], grid_resolution=60)
        x_vals = pdp_results['grid_values'][0]
        y_vals = pdp_results['average'][0]

        # 高斯平滑，消除随机森林特有的“锯齿感”
        y_vals_smooth = gaussian_filter1d(y_vals, sigma=1.2)

        # 主线与优雅的底部渐变填充
        ax.plot(x_vals, y_vals_smooth, color='#C0392B', linewidth=3.5, zorder=3)
        ax.fill_between(x_vals, y_vals_smooth, np.min(y_vals_smooth) - (np.max(y_vals_smooth)-np.min(y_vals_smooth))*0.5, 
                        color='#E74C3C', alpha=0.15, zorder=2)
        
        # 坐标轴与网格的调优
        ax.set_ylim(np.min(y_vals_smooth) * 0.95, np.max(y_vals_smooth) * 1.05)
        ax.set_xlabel(feature_dict[feature], fontproperties=my_font, fontsize=14, weight='bold')
        
        if idx % 3 == 0:
            ax.set_ylabel('PM$_{2.5}$ 边际响应浓度 ($\\mu g/m^3$)', fontproperties=my_font, fontsize=13)
            
        ax.tick_params(labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle("随机森林核心特征对 PM$_{2.5}$ 浓度的非线性物理响应曲线", 
                 fontproperties=my_font, fontsize=22, weight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = os.path.join(FIG_DIR, "RF_PDP_Physical_Response_Premium.png")
    plt.savefig(save_path, bbox_inches='tight', transparent=False)
    plt.close()

    print(f"\n🎉 精美 PDP 曲线绘制完成！\n 👉 请前往查看大图: {save_path}")

if __name__ == "__main__":
    main()