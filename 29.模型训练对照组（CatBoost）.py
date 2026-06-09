import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans  
import warnings
import chinese_calendar as conc
import joblib

# 引入 CatBoost
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 基础路径配置
# =============================================================================
BASE_DIR = "/home/wangzonghan/bisheshuju"
DATA_FILE = f"{BASE_DIR}/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"

# 定义并创建特定的模型与图表输出路径
CB_MODEL_DIR = os.path.join(BASE_DIR, "Results", "Models_CatBoost")
CB_FIG_DIR = os.path.join(BASE_DIR, "Results", "Figures_CatBoost")
RF_MODEL_DIR = os.path.join(BASE_DIR, "Results", "Models_随机森林")

os.makedirs(CB_MODEL_DIR, exist_ok=True)
os.makedirs(CB_FIG_DIR, exist_ok=True)

# 字体配置
FONT_PATH = f"{BASE_DIR}/fonts/SimHei.ttf"
my_font = FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else FontProperties()
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("="*70)
    print("🚀 启动 CatBoost 对照模型训练管线 ")
    print("="*70)

    # =============================================================================
    # 1. 加载数据与特征强制对齐
    # =============================================================================
    print("\n📂 [1/4] 加载数据集并同步随机森林特征池...")
    df = pd.read_parquet(DATA_FILE)
    
    # 时间特征补齐
    if 'month' not in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['season'] = (df['date'].dt.month % 12 // 3 + 1)
        df['is_holiday'] = df['date'].apply(lambda x: 1 if conc.is_holiday(x) else 0)

    # 随机森林目录读取特征清单，确保对比试验的控制变量
    rf_features_path = os.path.join(RF_MODEL_DIR, "best_features_list_terrain.pkl")
    if not os.path.exists(rf_features_path):
        print(f"❌ 找不到 RF 的特征列表，请检查路径: {rf_features_path}")
        return
    best_features = joblib.load(rf_features_path)
    print(f"  ✅ 已加载随机森林最优特征清单，共 {len(best_features)} 个特征。")

    # =============================================================================
    # 2. 空间分层抽样 
    # =============================================================================
    print("\n🌍 [2/4] 执行空间分层隔离抽样 (K-Means)...")
    sites_info = df[['site_code', 'lon', 'lat']].drop_duplicates().reset_index(drop=True)
    num_test = int(len(sites_info) * 0.15)
    
    kmeans = KMeans(n_clusters=num_test, random_state=42)
    sites_info['spatial_cluster'] = kmeans.fit_predict(sites_info[['lon', 'lat']])
    
    test_sites_df = sites_info.groupby('spatial_cluster').apply(
        lambda x: x.sample(n=1, random_state=42)
    ).reset_index(drop=True)
    
    test_sites = test_sites_df['site_code'].tolist()
    train_df = df[~df['site_code'].isin(test_sites)].copy()
    test_df = df[df['site_code'].isin(test_sites)].copy()

    X_train_full = train_df[best_features].astype('float32')
    y_train_full = train_df['pm25_hourly'].astype('float32')
    X_test = test_df[best_features].astype('float32')
    y_test = test_df['pm25_hourly'].astype('float32')

    # 为 CatBoost 设置早停验证集
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=42)

    # =============================================================================
    # 3. CatBoost 模型训练 
    # =============================================================================
    print("\n⚙️ [3/4] 启动 CatBoost 训练 ...")
    
    cb_model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.05,
        depth=8,
        loss_function='RMSE',
        eval_metric='R2',
        task_type='GPU',        
        random_seed=42,
        od_type='Iter',           # 开启早停机制
        od_wait=50                # 50 轮不进步则停止
    )
    
    cb_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=100
    )

    # =============================================================================
    # 4. 双尺度评估与结果持久化
    # =============================================================================
    print("\n" + "🔥"*30)
    print("  🎯 CatBoost 模型独立测试评估报告")
    print("🔥"*30)
    
    preds_hourly = cb_model.predict(X_test)
    r2_h = r2_score(y_test, preds_hourly)
    rmse_h = np.sqrt(mean_squared_error(y_test, preds_hourly))
    
    # 日均尺度聚合
    eval_df = test_df[['date', 'site_code', 'pm25_hourly']].copy()
    eval_df['pred_hourly'] = preds_hourly
    daily_eval = eval_df.groupby(['date', 'site_code']).mean().reset_index()
    r2_d = r2_score(daily_eval['pm25_hourly'], daily_eval['pred_hourly'])
    rmse_d = np.sqrt(mean_squared_error(daily_eval['pm25_hourly'], daily_eval['pred_hourly']))
    
    print(f"  🕒 小时级 -> R²: {r2_h:.4f} | RMSE: {rmse_h:.2f}")
    print(f"  📅 日均级 -> R²: {r2_d:.4f} | RMSE: {rmse_d:.2f}")

    # 保存模型至 Models_CatBoost 文件夹
    model_save_path = os.path.join(CB_MODEL_DIR, "best_cb_model_terrain.pkl")
    joblib.dump(cb_model, model_save_path)
    # 同步保存特征列表方便后续绘图脚本调用
    joblib.dump(best_features, os.path.join(CB_MODEL_DIR, "best_features_list_terrain.pkl"))
    print(f"\n✅ 最优模型已保存至: {model_save_path}")

    # =============================================================================
    # 5. 特征重要性可视化
    # =============================================================================
    print("\n📊 [4/4] 正在生成 CatBoost 特征重要性分析图...")
    importances = cb_model.get_feature_importance()
    df_imp = pd.DataFrame({'Feature': best_features, 'Importance': importances}).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 10), dpi=300)
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
    plt.title('CatBoost 特征重要性评估 (对照模型)', fontproperties=my_font, fontsize=16)
    plt.xlabel('贡献度', fontproperties=my_font, fontsize=14)
    plt.ylabel('特征变量', fontproperties=my_font, fontsize=14)
    plt.tight_layout()
    
    fig_save_path = os.path.join(CB_FIG_DIR, "CB_Feature_Importance_Terrain.png")
    plt.savefig(fig_save_path)
    plt.close()
    print(f"✅ 重要性分析图已保存至: {fig_save_path}")

if __name__ == "__main__":
    main()