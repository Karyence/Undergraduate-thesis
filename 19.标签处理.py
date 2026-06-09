import os
import zipfile
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 核心路径与研究区配置
# =============================================================================
ZIP_PATH = "/home/wangzonghan/Raw_Station_Archive/站点_20250101-20251231.zip"
SITE_LIST_PATH = "/data/Environment/Observation/全国空气质量/_站点列表/站点列表-2022.02.13起.csv"
OUTPUT_CSV = "/home/wangzonghan/bisheshuju/PM25_Labels/YRD_PM25_Hourly_2025_Sites.csv" 

# 长三角研究区边界
MIN_LON, MAX_LON = 114.5, 123.0
MIN_LAT, MAX_LAT = 27.0, 35.5

def process_ground_truth_with_coords():
    print("=" * 80)
    print("🚀 开始构建带有高精度经纬度的 PM2.5 真实标签集 ")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 第一步：解析站点列表，执行空间裁剪
    # -------------------------------------------------------------------------
    print("\n📌 [1/4] 解析站点列表，锁定长三角研究区内的监测站...")
    if not os.path.exists(SITE_LIST_PATH):
        print(f"❌ 找不到站点列表文件: {SITE_LIST_PATH}")
        return
        
    try:
        df_sites = pd.read_csv(SITE_LIST_PATH, encoding='utf-8')
    except:
        df_sites = pd.read_csv(SITE_LIST_PATH, encoding='gbk')
        
    site_col = next((c for c in df_sites.columns if '编码' in c or '编号' in c or '代码' in c), '监测点编码')
    lon_col = next((c for c in df_sites.columns if '经度' in c or 'lon' in c.lower()), '经度')
    lat_col = next((c for c in df_sites.columns if '纬度' in c or 'lat' in c.lower()), '纬度')
    
    df_sites = df_sites.rename(columns={site_col: 'site_code', lon_col: 'lon', lat_col: 'lat'})
    df_sites['lon'] = pd.to_numeric(df_sites['lon'], errors='coerce')
    df_sites['lat'] = pd.to_numeric(df_sites['lat'], errors='coerce')
    
    df_yrd_sites = df_sites[
        (df_sites['lon'] >= MIN_LON) & (df_sites['lon'] <= MAX_LON) & 
        (df_sites['lat'] >= MIN_LAT) & (df_sites['lat'] <= MAX_LAT)
    ].copy()
    
    valid_site_codes = set(df_yrd_sites['site_code'].unique())
    print(f"   -> 全国总计 {len(df_sites)} 个站点，长三角框内共保留 {len(valid_site_codes)} 个目标站点。")

    # -------------------------------------------------------------------------
    # 第二步：从 ZIP 压缩包中提取 2025 年 1-12 月的小时数据
    # -------------------------------------------------------------------------
    if not os.path.exists(ZIP_PATH):
        print(f"❌ 找不到压缩包: {ZIP_PATH}")
        return

    hourly_frames = [] 
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        csv_files = sorted([f for f in zf.namelist() if f.endswith('.csv')])
        target_files = [f for f in csv_files if "202501" <= f.split("_")[-1][:6] <= "202512"]
        
        print(f"\n📌 [2/4] 从压缩包提取 {len(target_files)} 个观测日 (全年)，执行融化与提纯...")
        
        for file in tqdm(target_files, desc="数据清洗进度"):
            try:
                with zf.open(file) as f:
                    try:
                        df = pd.read_csv(f, encoding='utf-8')
                    except:
                        f.seek(0)
                        df = pd.read_csv(f, encoding='gbk')
                
                df_pm25 = df[df['type'] == 'PM2.5']
                
                df_long = pd.melt(
                    df_pm25, 
                    id_vars=['date', 'hour', 'type'], 
                    var_name='site_code', 
                    value_name='pm25_hourly'
                )
                
                df_long = df_long[df_long['site_code'].isin(valid_site_codes)]
                if df_long.empty: continue
                    
                df_long['pm25_hourly'] = pd.to_numeric(df_long['pm25_hourly'], errors='coerce')
                df_long.loc[(df_long['pm25_hourly'] < 0) | (df_long['pm25_hourly'] > 1000), 'pm25_hourly'] = None
                df_long = df_long.dropna(subset=['pm25_hourly'])
                
                # 停止计算 groupby mean，直接保留 hour 字段并装入列表
                df_long['hour'] = df_long['hour'].astype(int)
                hourly_frames.append(df_long[['date', 'hour', 'site_code', 'pm25_hourly']])
                
            except Exception as e:
                pass

    # -------------------------------------------------------------------------
    # 第三步：时空联结 (Merge)
    # -------------------------------------------------------------------------
    print("\n📌 [3/4] 正在将小时浓度与站点经纬度进行空间联结...")
    final_pm25_df = pd.concat(hourly_frames, ignore_index=True)
    
    final_pm25_df['date'] = pd.to_datetime(final_pm25_df['date'].astype(str), format='%Y%m%d')
    
    final_merged_df = pd.merge(
        final_pm25_df, 
        df_yrd_sites[['site_code', 'lon', 'lat']], 
        on='site_code', 
        how='left'
    )

    # -------------------------------------------------------------------------
    # 第四步：导出结果
    # -------------------------------------------------------------------------
    print("\n📌 [4/4] 导出最终的完美标签集...")
    final_merged_df = final_merged_df[['date', 'hour', 'site_code', 'lon', 'lat', 'pm25_hourly']]
    final_merged_df = final_merged_df.sort_values(by=['date', 'site_code', 'hour']).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final_merged_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "=" * 80)
    print(f"✅ 极其完美的小时级 Y (目标变量) 生成完毕！")
    print(f"📊 最终标签样本数: {final_merged_df.shape[0]:,} 行")
    print(f"📍 覆盖长三角站点: {final_merged_df['site_code'].nunique()} 个")
    print(f"📋 数据头部预览:\n{final_merged_df.head()}")
    print(f"💾 已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    process_ground_truth_with_coords()