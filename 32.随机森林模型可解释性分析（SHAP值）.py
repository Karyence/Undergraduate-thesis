import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import shap
import warnings
import chinese_calendar as conc

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 基础路径配置与中文字体
# =============================================================================
BASE_DIR = "/home/wangzonghan/bisheshuju"
DATA_FILE = f"{BASE_DIR}/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"

RF_MODEL_DIR = f"{BASE_DIR}/Results/Models_随机森林"
FIG_DIR = f"{BASE_DIR}/Results/Figures_随机森林"
os.makedirs(FIG_DIR, exist_ok=True)

import matplotlib.font_manager as fm

FONT_PATH = f"{BASE_DIR}/fonts/SimHei.ttf"
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    my_font = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = my_font.get_name()
else:
    print(f"⚠️ 警告：找不到字体文件 {FONT_PATH}，中文可能显示异常！")

plt.rcParams['axes.unicode_minus'] = False 

def main():
    print("="*70)
    print("🚀 启动：33. 随机森林模型物理可解释性分析 ")
    print("="*70)

    # =============================================================================
    # 1. 加载模型与特征清单
    # =============================================================================
    print("\n📂 [1/4] 加载 Random Forest 最优模型与特征表...")
    rf_model = joblib.load(os.path.join(RF_MODEL_DIR, "best_rf_model_terrain.pkl"))
    best_features = joblib.load(os.path.join(RF_MODEL_DIR, "best_features_list_terrain.pkl"))

    feature_name_mapping = {
        'AOD': '气溶胶光学厚度 (AOD)',
        'month': '月份 (Month)',
        'season': '季节 (Season)',
        'hour': '小时 (Hour)',
        'day_of_week': '星期 (Day of week)',
        'is_holiday': '是否节假日 (Holiday)',
        'ERA5_BLH': '边界层高度 (BLH)',
        'ERA5_D2M': '2m露点温度 (D2m)',
        'ERA5_T2M': '2m温度 (T2m)',
        'ERA5_TP': '累计降水 (TP)',
        'ERA5_WIND': '风速 (Wind)',
        'DEM': '高程 (DEM)',
        'Slope': '坡度 (Slope)',
        'NDVI': '植被指数 (NDVI)',
        'LC_cropland_frac': '耕地占比 (Cropland)',
        'LC_forest_frac': '森林占比 (Forest)',
        'LC_grassland_frac': '草地占比 (Grassland)',
        'LC_barren_frac': '裸地占比 (Barren)',
        'LC_building_frac': '建筑占比 (Building)',
        'LC_traffic_frac': '交通占比 (Traffic)',
        'is_weekend': '周末 (Weekend)'
    }
    
    display_feature_names = [feature_name_mapping.get(f, f) for f in best_features]

    # =============================================================================
    # 2. 加载数据并提取 15% 测试集
    # =============================================================================
    print("⏳ [2/4] 加载数据集并提取独立测试集...")
    df = pd.read_parquet(DATA_FILE)
    
    if 'month' not in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['season'] = (df['date'].dt.month % 12 // 3 + 1)
        df['is_holiday'] = df['date'].apply(lambda x: 1 if conc.is_holiday(x) else 0)

    from sklearn.cluster import KMeans
    sites_info = df[['site_code', 'lon', 'lat']].drop_duplicates().reset_index(drop=True)
    num_test = int(len(sites_info) * 0.15)
    kmeans = KMeans(n_clusters=num_test, random_state=42)
    sites_info['spatial_cluster'] = kmeans.fit_predict(sites_info[['lon', 'lat']])
    
    test_sites_df = sites_info.groupby('spatial_cluster').apply(lambda x: x.sample(n=1, random_state=42)).reset_index(drop=True)
    test_df = df[df['site_code'].isin(test_sites_df['site_code'].tolist())].copy()

    print("🎯 [3/4] 随机抽取 3000 个样本进入 TreeExplainer 解析引擎...")
    X_test = test_df[best_features].astype('float32')
    X_sample = X_test.sample(n=3000, random_state=42)
    
    X_sample.columns = display_feature_names

    # =============================================================================
    # 3. 计算 SHAP 值 
    # =============================================================================
    print("⚙️ 检测到 cuML(GPU) 模型，自动切换为通用黑盒解释器 KernelExplainer...")

    # 构造一个极其健壮的预测包装函数，负责抹平 CPU/GPU 和数据格式差异
    def safe_predict(data):
        preds = rf_model.predict(np.array(data, dtype=np.float32))
        if hasattr(preds, 'get'): 
            return preds.get()
        if hasattr(preds, 'to_numpy'):
            return preds.to_numpy()
        return np.array(preds)

    background = shap.kmeans(X_sample, 50)
    
    explainer = shap.KernelExplainer(safe_predict, background)

    print("⏳ 正在进行 SHAP 归因计算 (处理 3000 个样本)，这需要几分钟，请关注下方进度条...")
    # silent=False 会在终端打印一个进度条，让你清楚看到计算进度
    shap_values = explainer.shap_values(X_sample, silent=False)

    # =============================================================================
    # 4. 绘制纯净版 SHAP 蜂之图
    # =============================================================================
    print("🎨 [4/4] 正在渲染高分辨率 SHAP 特征贡献分布图...")
    
    # 创建适合单图的画布尺寸 (宽10，高8)
    fig = plt.figure(figsize=(10, 8), dpi=300)
    
    # 绘制 SHAP summary plot
    shap.summary_plot(shap_values, X_sample, show=False, plot_type="dot", 
                      color_bar_label="特征数值大小 (Feature Value)")
    
    # 获取当前坐标轴进行定制化美化
    ax = plt.gca()
    ax.set_xlabel('SHAP值 (对 PM$_{2.5}$ 预测的贡献量 $\mu g/m^3$)', fontsize=16, weight='bold', labelpad=15)
    ax.set_title('随机森林模型反演 PM$_{2.5}$ 机制解析 (SHAP特征贡献分布)', fontsize=18, weight='bold', pad=20)
    
    # 加粗和放大坐标轴刻度字体
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=14)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存出图
    save_path = os.path.join(FIG_DIR, "RF_Interpretability_SHAP_Summary.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print("="*70)
    print(f"🎉 完美！单幅纯净版 SHAP 物理可解释性图表已生成！")
    print(f"👉 图片路径: {save_path}")
    print("="*70)

if __name__ == "__main__":
    main()