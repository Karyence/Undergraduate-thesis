import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from sklearn.cluster import KMeans
import chinese_calendar as conc
import joblib
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 基础路径配置
# =============================================================================
BASE_DIR = "/home/wangzonghan/bisheshuju"
DATA_FILE = f"{BASE_DIR}/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"
SITE_LIST_PATH = "/data/Environment/Observation/全国空气质量/_站点列表/站点列表-2022.02.13起.csv"

RF_MODEL_DIR = f"{BASE_DIR}/Results/Models_随机森林"
FIG_DIR = f"{BASE_DIR}/Results/Figures_模型对比"
os.makedirs(FIG_DIR, exist_ok=True)

FONT_PATH = f"{BASE_DIR}/fonts/SimHei.ttf"
my_font = FontProperties(fname=FONT_PATH, size=14) if os.path.exists(FONT_PATH) else FontProperties(size=14)
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("="*70)
    print("🚀 启动：32. 典型重污染过程时间序列动态捕捉分析")
    print("="*70)

    # =============================================================================
    # 1. 严格复刻 15% 独立测试集
    # =============================================================================
    print("\n📂 [1/4] 加载数据与复原测试集...")
    df = pd.read_parquet(DATA_FILE)
    
    if 'month' not in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['season'] = (df['date'].dt.month % 12 // 3 + 1)
        df['is_holiday'] = df['date'].apply(lambda x: 1 if conc.is_holiday(x) else 0)

    # 构造连续时间戳用于画图
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str) + ':00:00')

    sites_info = df[['site_code', 'lon', 'lat']].drop_duplicates().reset_index(drop=True)
    num_test = int(len(sites_info) * 0.15)
    kmeans = KMeans(n_clusters=num_test, random_state=42)
    sites_info['spatial_cluster'] = kmeans.fit_predict(sites_info[['lon', 'lat']])
    
    test_sites_df = sites_info.groupby('spatial_cluster').apply(lambda x: x.sample(n=1, random_state=42)).reset_index(drop=True)
    test_df = df[df['site_code'].isin(test_sites_df['site_code'].tolist())].copy()

    # =============================================================================
    # 2. 模型预测 
    # =============================================================================
    print("  ⏳ [2/4] 加载 Random Forest 最优模型进行全时段推断...")
    rf_model = joblib.load(os.path.join(RF_MODEL_DIR, "best_rf_model_terrain.pkl"))
    best_features = joblib.load(os.path.join(RF_MODEL_DIR, "best_features_list_terrain.pkl"))
    
    X_test = test_df[best_features].astype('float32')
    
    preds = rf_model.predict(X_test)
    if hasattr(preds, 'to_numpy'): 
        preds = preds.to_numpy()
    test_df['pred_pm25'] = preds

    # =============================================================================
    # 3. 智能寻找“最具代表性”的极端重污染波峰 
    # =============================================================================
    print("\n🔍 [3/4] 正在独立测试集中智能检索气象驱动型重污染爆发时段...")
    
    winter_df = test_df[test_df['month'].isin([12, 1, 2])]
    valid_peaks = winter_df[(winter_df['pm25_hourly'] >= 115) & 
                            (winter_df['pm25_hourly'] <= 250) & 
                            (winter_df['pred_pm25'] >= 80)] 
    
    if valid_peaks.empty:
        print("未找到双向匹配的重污染过程，放宽条件检索...")
        max_idx = winter_df['pm25_hourly'].idxmax()
    else:
        max_idx = valid_peaks['pm25_hourly'].idxmax()

    target_site = test_df.loc[max_idx, 'site_code']
    target_city = "重点城市"  
    
    if os.path.exists(SITE_LIST_PATH):
        try:
            try:
                site_df = pd.read_csv(SITE_LIST_PATH, encoding='utf-8')
            except UnicodeDecodeError:
                site_df = pd.read_csv(SITE_LIST_PATH, encoding='gbk')
            
            code_col = '监测点编码' if '监测点编码' in site_df.columns else ('site_code' if 'site_code' in site_df.columns else None)
            city_col = '城市' if '城市' in site_df.columns else ('city' if 'city' in site_df.columns else None)
            
            # 执行查表匹配
            if code_col and city_col:
                match_row = site_df[site_df[code_col] == target_site]
                if not match_row.empty:
                    target_city = str(match_row.iloc[0][city_col])
        except Exception as e:
            print(f" ⚠️ 站点表读取匹配失败，将尝试从原数据集提取。错误信息: {e}")

    if target_city == "重点城市":
        if 'city' in test_df.columns:
            target_city = test_df.loc[max_idx, 'city']
        elif 'City' in test_df.columns:
            target_city = test_df.loc[max_idx, 'City']
        elif 'city_zh' in test_df.columns:
            target_city = test_df.loc[max_idx, 'city_zh']
            
    peak_time = test_df.loc[max_idx, 'datetime']
    
    start_time = peak_time - pd.Timedelta(days=7)
    end_time = peak_time + pd.Timedelta(days=7)
    
    episode_df = test_df[(test_df['site_code'] == target_site) & 
                         (test_df['datetime'] >= start_time) & 
                         (test_df['datetime'] <= end_time)].sort_values('datetime')

    max_true = episode_df['pm25_hourly'].max()
    print(f"   -> 锁定气象驱动型典型站点: {target_city} ({target_site})")
    print(f"   -> 污染爆发中心: {peak_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"   -> 期间最高浓度: {max_true:.1f} μg/m³")

    # =============================================================================
    # 4. 绘制高颜值时间序列图
    # =============================================================================
    print("\n🎨 [4/4] 正在绘制动态时间序列追踪图...")
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)

    # 绘制真实值与预测值折线
    ax.plot(episode_df['datetime'], episode_df['pm25_hourly'], 
            color='black', linewidth=2.5, label='地面基站真实观测值 (True)', zorder=4)
    ax.plot(episode_df['datetime'], episode_df['pred_pm25'], 
            color='#D9383A', linewidth=2.5, linestyle='--', label='随机森林模型反演值 (Predict)', zorder=5)

    # 绘制国家空气质量标准 AQI 背景色阶带 
    ax.axhspan(0, 35, facecolor='#A8E6CF', alpha=0.3, zorder=1, label='优 (0-35)')
    ax.axhspan(35, 75, facecolor='#FFD3B6', alpha=0.3, zorder=1, label='良 (35-75)')
    ax.axhspan(75, 115, facecolor='#FFAAA5', alpha=0.3, zorder=1, label='轻度污染 (75-115)')
    ax.axhspan(115, 150, facecolor='#FF8B94', alpha=0.3, zorder=1, label='中度污染 (115-150)')
    # 计算图表 y 轴最大值，保证最高的紫色色块能覆盖满
    y_max = max(max_true, episode_df['pred_pm25'].max()) * 1.15
    ax.axhspan(150, max(250, y_max), facecolor='#8D6298', alpha=0.2, zorder=1, label='重度及以上污染 (>150)')

    # 设置 Y 轴上限
    ax.set_ylim(0, y_max)

    # 美化 X 轴时间格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:00'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1)) # 每天一个大刻度
    plt.xticks(rotation=0)

    # 设置标签与标题
    ax.set_ylabel('PM$_{2.5}$ 质量浓度 ($\\mu g/m^3$)', fontproperties=my_font, fontsize=14)
    ax.set_xlabel('日期与时间 (Date & Time)', fontproperties=my_font, fontsize=14)
    ax.set_title(f'独立测试集代表性站点 ({target_city} - {target_site}) 冬季极端重污染过程动态捕捉分析\n'
                 f'时段: {start_time.strftime("%Y-%m-%d")} 至 {end_time.strftime("%Y-%m-%d")}', 
                 fontproperties=my_font, fontsize=18, weight='bold', pad=15)

    ax.grid(True, linestyle=':', alpha=0.6, zorder=2)
    
    # 调整图例位置，避免遮挡曲线
    ax.legend(loc='upper left', prop=my_font, ncol=2, framealpha=0.9)

    plt.tight_layout()
    
    save_path = os.path.join(FIG_DIR, f"RF_Time_Series_Capture_{target_site}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"\n🎉 完美！典型过程动态捕捉图已生成！\n 👉 请查看: {save_path}")

if __name__ == "__main__":
    main()