import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 核心路径配置 
# =============================================================================
FILE_PATH = "/home/wangzonghan/bisheshuju/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"

def verify_ml_dataset():
    print("=" * 80)
    print("🔬 机器学习最终数据集 (Hourly Parquet) 深度体检报告")
    print("=" * 80)

    # 加载数据集
    if not os.path.exists(FILE_PATH):
        print(f"❌ 找不到训练集文件: {FILE_PATH}")
        return
        
    df = pd.read_parquet(FILE_PATH)
    
    print(f"✅ 成功加载数据！")
    print(f"📊 矩阵维度: {df.shape[0]:,} 行, {df.shape[1]} 列")
    
    # -------------------------------------------------------------------------
    # 步骤 1: 缺失值 (NaN) 扫描
    # -------------------------------------------------------------------------
    print("\n📌 [1/4] 缺失值扫描:")
    total_nans = df.isna().sum().sum()
    if total_nans == 0:
        print("   -> 🏆 完美！全表没有任何缺失值，可以直接喂给模型。")
    else:
        print(f"   -> ⚠️ 警告：发现 {total_nans} 个缺失值，请检查特征对齐环节。")

    # -------------------------------------------------------------------------
    # 🌟 步骤 2: 空间站点与时间存活扫描 
    # -------------------------------------------------------------------------
    print("\n📌 [2/4] 空间站点与时序存活扫描:")
    valid_sites = df['site_code'].nunique()
    total_days = df['date'].nunique()
    total_hours = df['hour'].nunique() 
    
    print(f"   -> 📍 最终存活的高质量【全维特征站点】数: {valid_sites} 个")
    print(f"   -> 📅 包含的有效观测天数: {total_days} 天")
    print(f"   -> ⏰ 包含的有效观测小时跨度: {total_hours} 个 (预期为 24)")
    
    if valid_sites == 298:
        print("   -> 🎯 验证通过：与之前的 298 个站点完美吻合 (40个海洋/越界站点被持续物理隔离)！")

    # -------------------------------------------------------------------------
    # 步骤 3: 特征数据类型扫描
    # -------------------------------------------------------------------------
    print("\n📌 [3/4] 核心特征数据类型扫描:")
    print("   -> 确保送入模型的特征全部为数值型 (float/int):")
    exclude_cols = ['date', 'site_code', 'lon', 'lat']
    features = [c for c in df.columns if c not in exclude_cols]
    print(df[features].dtypes.value_counts())

    # -------------------------------------------------------------------------
    # 步骤 4: 物理常识与相关性分析
    # -------------------------------------------------------------------------
    print("\n📌 [4/4] 核心特征与 PM2.5 (小时级) 的相关性 (Pearson Correlation) 探针:")
    print("   -> 正数表示正相关，负数表示负相关")
    
    # 计算 Pearson 相关系数 (目标变量更新为 pm25_hourly)
    corr_matrix = df[features].corr()
    pm25_corr = corr_matrix['pm25_hourly'].sort_values(ascending=False)
    
    for feature, corr_val in pm25_corr.items():
        if feature == 'pm25_hourly':
            continue
        icon = "🔴 正相关" if corr_val > 0 else "🟢 负相关"
        print(f"   - {feature:<18} : {corr_val:>6.3f}  ({icon})")

    print("\n" + "=" * 80)
    print("💡 结果解读提示：")
    print("1. AOD (气溶胶光学厚度) 应呈现显著正相关。")
    print("2. ERA5_BLH (边界层高度) 与 ERA5_WIND (风速) 必须呈现负相关（核心扩散条件）。")
    print("3. 如果相关性方向符合上述物理规律，恭喜你，可以开始训练模型了！")

if __name__ == "__main__":
    verify_ml_dataset()