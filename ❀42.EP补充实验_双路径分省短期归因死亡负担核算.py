# -*- coding: utf-8 -*-
"""
===============================================================================
42. EP 补充实验：双路径 1-km 逐日 PM2.5 短期可归因死亡负担与不确定性核算（分省版）
===============================================================================
用途：
  为《Environmental Pollution》修改稿 §3.3 (Health Implications for Mortality Burden)
  提供严谨、可复现的定量健康影响评估 (Health Impact Assessment, HIA) 结果。

核心方法（控制变量 HIA）：
  固定同一套 1-km 人口分布 (WorldPop 2025)、同一套 2025 年官方分省常住人口粗死亡率
  与同一套已发表短期暴露-反应函数 (ERF)，仅替换“小时训练-日均聚合 (Hourly-to-Daily)”
  与“直接日均训练 (Direct-Daily)”两套 2025 年逐日 1-km PM2.5 网格暴露场，定量评估
  反演路径偏差向短期 PM2.5 可归因死亡负担的传递效应：
      C_lag01(i, d) = 0.5 * [C(i, d) + C(i, d-1)]      # 2日滑动平均 (lag 0-1)
      beta          = ln(1 + ER / 100) / 10.0          # 每升高 10 ug/m3 的对数风险斜率
      RR(i, d)      = exp(beta * max(0, C_lag01 - C0)) # 相对风险
      AF(i, d)      = (RR(i, d) - 1) / RR(i, d)        # 归因分数
      Y_daily(i)    = POP(i) * I0_prov(i) / 365.0      # 网格日均基线死亡人数
      M(i, d)       = Y_daily(i) * AF(i, d)            # 网格当日可归因死亡人数

参数出处（均已核验）：
  1. 2025 年分省常住人口粗死亡率 (‰)：来自各省 2025 年国民经济和社会发展统计公报
     - 上海市: 6.61‰ | 江苏省: 7.80‰ | 浙江省: 6.54‰ | 安徽省: 8.67‰
     - 江西省: 7.51‰ | 河南省: 8.39‰ | 山东省: 8.21‰ | 福建省: 6.89‰
     - 湖北省: 8.86‰ | 全国兜底 (仅用于海岸线极少数未落入省界多边形的零星陆地格点): 8.04‰
  2. 短期暴露-反应系数 (lag 0-1, 每 10 ug/m3 PM2.5)：
     - Liu et al. (2019), N. Engl. J. Med. 381, 705-715 (652 cities, 全因日死亡):
       ER = 0.68% (95% CI: 0.59% - 0.77%) [PMC7891185 原文核实，与统计公报全因死亡率 1:1 匹配]
     - Chen et al. (2017), Am. J. Respir. Crit. Care Med. 196, 73-81 (中国 272 城, 总非意外死亡):
       ER = 0.22% [PubMed 28248546 核实]。
       【注意】：请于运行前在浙大校园网打开 Chen et al. (2017) 原文 Table 1，将 0.22% 对应的
       95% 后验区间 (95% PI) 填入下方 CHEN_2017_CI_LOW 和 CHEN_2017_CI_HIGH；若暂未填入，
       脚本将安全输出 Chen et al. (2017) 的精确点估计而不编造区间。
  3. 反事实阈值 C0：
     - WHO_15 : 15.0 ug/m3 (WHO 2021 短期日均指导值，主情景)
     - GB_75  : 75.0 ug/m3 (中国 GB 3095-2012 二级日均标准，超标污染日情景)
     - Zero_0 :  0.0 ug/m3 (无阈值基准敏感性情景)

输出文件（保存在 BASE_DIR/Results/EP_Supplementary_Stats/）：
  - EP_Mortality_HIA_Summary.json            : 全区汇总（全年/冬季/各 ERF/各阈值/95% UI）
  - EP_Mortality_HIA_Provincial_Summary.csv  : 分省汇总表（人口、死亡率、双路径死亡数、差值、每10万人负担）
  - EP_Mortality_HIA_Monthly_TimeSeries.csv  : 逐月双路径可归因死亡人数时间序列
  - Fig_EP_Mortality_Burden_Comparison.png   : §3.3 出版级双面板对比图 (300 dpi)
===============================================================================
"""

import os
import glob
import gc
import json
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import joblib
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.prepared import prep

try:
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

try:
    from shapely import vectorized as shp_vec
    HAS_SHP_VEC = True
except ImportError:
    HAS_SHP_VEC = False

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif", "Liberation Serif"]
plt.rcParams["axes.unicode_minus"] = False

# =============================================================================
# 0. 基础路径配置（与 41 号脚本保持 100% 一致）
# =============================================================================
BASE_DIR = "/home/wangzonghan/bisheshuju"
INFERENCE_DIR = "/data/Machine_Learning/wangzonghan/YRD_PM25_Project/Inference_Features"
POP_NC_PATH = f"{BASE_DIR}/Population/YRD_Population_1km_2025.nc"
LC_NC_PATH = f"{BASE_DIR}/LandCover/YRD_LandCover_Fractions_1km_Final.nc"

HOURLY_MODEL_PATH = f"{BASE_DIR}/Results/Models_随机森林/best_rf_model_terrain.pkl"
HOURLY_FEAT_PATH = f"{BASE_DIR}/Results/Models_随机森林/best_features_list_terrain.pkl"

DAILY_MODEL_PATH = f"{BASE_DIR}/Results/Models_对照组_日均RF/best_rf_daily_model.pkl"
DAILY_FEAT_PATH = f"{BASE_DIR}/Results/Models_对照组_日均RF/best_features_list_daily.pkl"

PROVINCE_SHP_PATH = "/home/yanchengzhu/Map/GIS/中国_省.shp"

OUTPUT_DIR = f"{BASE_DIR}/Results/EP_Supplementary_Stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. 公开权威参数配置（2025 年分省粗死亡率 + 短期 ERF 参数 + 阈值）
# =============================================================================
# 2025 年各省常住人口粗死亡率（单位：‰，除以 1000 后为人·年死亡率）
PROVINCE_MORTALITY_RATE_PER_1000 = {
    "上海": 6.61,
    "江苏": 7.80,
    "浙江": 6.54,
    "安徽": 8.67,
    "江西": 7.51,
    "河南": 8.39,
    "山东": 8.21,
    "福建": 6.89,
    "湖北": 8.86,
}
PROVINCE_EN_NAME = {
    "上海": "Shanghai",
    "江苏": "Jiangsu",
    "浙江": "Zhejiang",
    "安徽": "Anhui",
    "江西": "Jiangxi",
    "河南": "Henan",
    "山东": "Shandong",
    "福建": "Fujian",
    "湖北": "Hubei",
    "其他/海岸边缘": "Other/Coastal",
}
NATIONAL_FALLBACK_RATE_PER_1000 = 8.04  # 国家统计局 2025 年全国粗死亡率（仅作极少数未匹配边缘格点兜底）

# 请在浙大内网核对 Chen et al. (2017) 原文 Table 1 中 0.22% 的 95% PI 后填入下面两行（例如 0.15, 0.28）
CHEN_2017_CI_LOW = None   # 例如填 0.15（需核对原文 Table 1）
CHEN_2017_CI_HIGH = None  # 例如填 0.28（需核对原文 Table 1）

ERF_SPECS = {
    "Liu_2019_AllCause": {
        "citation": "Liu et al. (2019), NEJM 381, 705-715 (652 cities, all-cause, lag 0-1)",
        "er_pct": 0.68,
        "er_low_pct": 0.59,
        "er_high_pct": 0.77,
    },
    "Chen_2017_China272": {
        "citation": "Chen et al. (2017), AJRCCM 196, 73-81 (272 Chinese cities, non-accidental, lag 0-1)",
        "er_pct": 0.22,
        "er_low_pct": CHEN_2017_CI_LOW,
        "er_high_pct": CHEN_2017_CI_HIGH,
    },
}

THRESHOLDS_UGM3 = {
    "WHO_15": 15.0,   # WHO 2021 24-h AQG（主情景）
    "GB_75": 75.0,    # GB 3095-2012 二级日均标准（超标污染日情景）
    "Zero_0": 0.0,    # 无阈值线性基准
}


# =============================================================================
# 2. 工具函数
# =============================================================================
def er_pct_to_beta(er_pct):
    """每升高 10 ug/m3 的百分比增量 ER(%) 转换为对数风险斜率 beta (per 1 ug/m3)。"""
    if er_pct is None:
        return None
    return float(np.log(1.0 + er_pct / 100.0) / 10.0)


def calc_af(conc_lag01, beta, c0):
    """计算归因分数 AF = (RR - 1) / RR，其中 RR = exp(beta * max(0, C - C0))。"""
    excess = np.maximum(0.0, conc_lag01 - c0)
    rr = np.exp(beta * excess)
    return (rr - 1.0) / rr


def batch_predict(model, X, batch_size=250000):
    preds = []
    for i in range(0, X.shape[0], batch_size):
        batch_pred = model.predict(X[i : i + batch_size])
        if hasattr(batch_pred, "get"):
            batch_pred = batch_pred.get()
        elif hasattr(batch_pred, "to_numpy"):
            batch_pred = batch_pred.to_numpy()
        preds.append(batch_pred)
    return np.concatenate(preds)


def load_static_missing_lc_features(missing_cols):
    if not missing_cols or not os.path.exists(LC_NC_PATH):
        return None
    with xr.open_dataset(LC_NC_PATH) as ds:
        lc_df = ds.to_dataframe().reset_index().rename(columns={"longitude": "lon", "latitude": "lat"})
        lc_raw_cols = [c for c in lc_df.columns if c not in ["lon", "lat", "time", "date", "spatial_ref", "x", "y"]]
        lc_df = lc_df.rename(columns={c: f"LC_{c}" for c in lc_raw_cols})
        keep_cols = [c for c in missing_cols if c in lc_df.columns]
        if not keep_cols:
            return None
        lc_sub = lc_df[["lon", "lat"] + keep_cols].copy()
        lc_sub["lon"] = lc_sub["lon"].round(3)
        lc_sub["lat"] = lc_sub["lat"].round(3)
        for c in keep_cols:
            lc_sub[c] = lc_sub[c].fillna(0.0).astype("float32")
        lc_sub = lc_sub.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
        return lc_sub


def match_province_key(raw_name):
    if raw_name is None:
        return None
    s = str(raw_name)
    for k in PROVINCE_MORTALITY_RATE_PER_1000:
        if k in s:
            return k
    return None


def assign_province_mortality(lons, lats, pop_aligned):
    """
    利用已有省界矢量将每个 1-km 网格匹配至对应省份及其 2025 年官方粗死亡率。
    仅对有效人口格点 (pop > 0) 做空间包含判断，兼顾精度与速度。
    """
    n_cells = len(lons)
    prov_labels = np.full(n_cells, "其他/海岸边缘", dtype=object)
    mort_rates = np.full(n_cells, NATIONAL_FALLBACK_RATE_PER_1000 / 1000.0, dtype=np.float64)

    if not os.path.exists(PROVINCE_SHP_PATH):
        print(f"⚠️ 未找到省界矢量 {PROVINCE_SHP_PATH}，将回退使用全国 2025 年粗死亡率 {NATIONAL_FALLBACK_RATE_PER_1000}‰。")
        return prov_labels, mort_rates

    reader = shpreader.Reader(PROVINCE_SHP_PATH)
    valid_idx = np.where(pop_aligned > 0)[0]
    valid_lons = lons[valid_idx]
    valid_lats = lats[valid_idx]

    matched_mask = np.zeros(len(valid_idx), dtype=bool)
    for rec in reader.records():
        attrs = rec.attributes
        # 自动探测省份名称字段
        pkey = None
        for val in attrs.values():
            pkey = match_province_key(val)
            if pkey is not None:
                break
        if pkey is None:
            continue

        geom = rec.geometry
        minx, miny, maxx, maxy = geom.bounds
        bbox_mask = (
            (~matched_mask)
            & (valid_lons >= minx)
            & (valid_lons <= maxx)
            & (valid_lats >= miny)
            & (valid_lats <= maxy)
        )
        cand_pos = np.where(bbox_mask)[0]
        if len(cand_pos) == 0:
            continue

        if HAS_SHP_VEC:
            inside = shp_vec.contains(geom, valid_lons[cand_pos], valid_lats[cand_pos])
        else:
            pgeom = prep(geom)
            inside = np.array(
                [pgeom.contains(Point(x, y)) for x, y in zip(valid_lons[cand_pos], valid_lats[cand_pos])],
                dtype=bool,
            )

        hit_pos = cand_pos[inside]
        if len(hit_pos) > 0:
            orig_hit_idx = valid_idx[hit_pos]
            prov_labels[orig_hit_idx] = pkey
            mort_rates[orig_hit_idx] = PROVINCE_MORTALITY_RATE_PER_1000[pkey] / 1000.0
            matched_mask[hit_pos] = True

    total_pop = float(np.sum(pop_aligned[valid_idx]))
    fallback_pop = float(np.sum(pop_aligned[valid_idx[~matched_mask]]))
    fallback_pct = fallback_pop / max(total_pop, 1.0) * 100.0
    print(f"  ✅ 分省死亡率空间匹配完成：匹配人口占比 {100.0 - fallback_pct:.2f}% | 海岸边缘兜底占比 {fallback_pct:.2f}%")
    return prov_labels, mort_rates


# =============================================================================
# 3. 主计算流程
# =============================================================================
def main():
    print("=" * 88)
    print("🚀 启动 EP 42号补充实验：双路径 1-km 逐日 PM2.5 分省短期可归因死亡负担核算")
    print("=" * 88)

    hr_model = joblib.load(HOURLY_MODEL_PATH)
    hr_feats = list(joblib.load(HOURLY_FEAT_PATH))
    dy_model = joblib.load(DAILY_MODEL_PATH)
    dy_feats = list(joblib.load(DAILY_FEAT_PATH))

    # 加载 WorldPop 1-km 人口
    ds_pop = xr.open_dataset(POP_NC_PATH)
    pop_df = ds_pop.to_dataframe().reset_index().rename(columns={"longitude": "lon", "latitude": "lat"})
    pop_col = [c for c in pop_df.columns if c not in ["lon", "lat", "spatial_ref"]][0]
    pop_df = pop_df[["lon", "lat", pop_col]].rename(columns={pop_col: "pop"})
    pop_df["lon"] = pop_df["lon"].round(3)
    pop_df["lat"] = pop_df["lat"].round(3)
    pop_df = pop_df.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)

    grid_files = sorted(glob.glob(f"{INFERENCE_DIR}/YRD_Grid_Hourly_Features_Terrain_2025*.parquet"))
    if not grid_files:
        raise FileNotFoundError(f"未找到网格推理特征文件: {INFERENCE_DIR}")

    all_parquet_cols = pd.read_parquet(grid_files[0]).columns.tolist()
    missing_dy_cols = [c for c in dy_feats if c not in all_parquet_cols]
    present_dy_cols = [c for c in dy_feats if c in all_parquet_cols]
    static_lc_patch = load_static_missing_lc_features(missing_dy_cols) if missing_dy_cols else None

    # 预构建 ERF 斜率字典（点估计 + 95% CI 上下界）
    beta_table = {}
    for erf_name, spec in ERF_SPECS.items():
        beta_table[erf_name] = {
            "mid": er_pct_to_beta(spec["er_pct"]),
            "low": er_pct_to_beta(spec["er_low_pct"]),
            "high": er_pct_to_beta(spec["er_high_pct"]),
        }

    # 累加器初始化占位
    grid_coords = None
    pop_aligned = None
    prov_labels = None
    daily_base_deaths = None  # Y_daily(i) = POP(i) * I0_prov(i) / 365
    aligned_missing_arrays = {}

    prev_h = None
    prev_d = None

    # 结构：acc[erf_name][thr_name][bound] -> dict(annual_h, annual_d, winter_h, winter_d)
    acc_total = {}
    acc_monthly = {}   # [erf_name][thr_name][month_int] -> dict(h_mid, d_mid, h_low, d_low, h_high, d_high)
    acc_prov = {}      # [erf_name][thr_name][prov_key]  -> dict(h_mid, d_mid, h_low, d_low, h_high, d_high, win_h_mid, win_d_mid)
    unique_provs = []

    for idx, fpath in enumerate(grid_files):
        date_str = os.path.basename(fpath).split("_")[-1].split(".")[0]
        month = int(date_str[4:6])
        is_winter = month in [12, 1, 2]

        gdf = pd.read_parquet(fpath)
        gdf["lon"] = gdf["lon"].round(3)
        gdf["lat"] = gdf["lat"].round(3)

        for f_col in hr_feats:
            if f_col not in gdf.columns:
                gdf[f_col] = 0.0
        X_hr = gdf[hr_feats].astype("float32").values
        gdf["pred_h"] = np.clip(batch_predict(hr_model, X_hr), 0.0, 300.0)

        agg_cols = ["pred_h"] + [c for c in present_dy_cols if c != "pred_h"]
        if HAS_CUDF:
            cgdf = cudf.from_pandas(gdf[["lat", "lon"] + agg_cols])
            daily_merged = cgdf.groupby(["lat", "lon"], sort=True)[agg_cols].mean().reset_index().to_pandas()
            del cgdf
        else:
            daily_merged = gdf.groupby(["lat", "lon"], sort=True)[agg_cols].mean().reset_index()

        # 首日初始化坐标、人口、分省死亡基线及所有累加器
        if grid_coords is None:
            grid_coords = daily_merged[["lon", "lat"]].copy()
            merged_pop = pd.merge(grid_coords, pop_df, on=["lon", "lat"], how="left")
            pop_aligned = np.maximum(np.nan_to_num(merged_pop["pop"].values, nan=0.0), 0.0)

            prov_labels, mort_rates = assign_province_mortality(
                grid_coords["lon"].values, grid_coords["lat"].values, pop_aligned
            )
            daily_base_deaths = pop_aligned * mort_rates / 365.0

            if static_lc_patch is not None:
                merged_lc = pd.merge(grid_coords, static_lc_patch, on=["lon", "lat"], how="left")
                for mc in missing_dy_cols:
                    aligned_missing_arrays[mc] = (
                        np.nan_to_num(merged_lc[mc].values, nan=0.0).astype("float32")
                        if mc in merged_lc.columns
                        else np.zeros(len(grid_coords), dtype="float32")
                    )
            else:
                for mc in missing_dy_cols:
                    aligned_missing_arrays[mc] = np.zeros(len(grid_coords), dtype="float32")

            unique_provs = sorted(set(prov_labels[pop_aligned > 0]))
            prov_masks = {pk: (prov_labels == pk) & (pop_aligned > 0) for pk in unique_provs}

            for erf_name in ERF_SPECS:
                acc_total[erf_name] = {}
                acc_monthly[erf_name] = {}
                acc_prov[erf_name] = {}
                for thr_name in THRESHOLDS_UGM3:
                    acc_total[erf_name][thr_name] = {
                        b: {"annual_h": 0.0, "annual_d": 0.0, "winter_h": 0.0, "winter_d": 0.0}
                        for b in ["mid", "low", "high"]
                    }
                    acc_monthly[erf_name][thr_name] = {
                        m: {"h_mid": 0.0, "d_mid": 0.0, "h_low": 0.0, "d_low": 0.0, "h_high": 0.0, "d_high": 0.0}
                        for m in range(1, 13)
                    }
                    acc_prov[erf_name][thr_name] = {
                        pk: {
                            "h_mid": 0.0, "d_mid": 0.0, "h_low": 0.0, "d_low": 0.0, "h_high": 0.0, "d_high": 0.0,
                            "win_h_mid": 0.0, "win_d_mid": 0.0,
                        }
                        for pk in unique_provs
                    }

        for mc in missing_dy_cols:
            daily_merged[mc] = aligned_missing_arrays[mc]
        X_dy = daily_merged[dy_feats].astype("float32").values
        val_d = np.clip(batch_predict(dy_model, X_dy), 0.0, 300.0)
        val_h = daily_merged["pred_h"].values

        # 计算 2 日滑动平均 (lag 0-1)；首日取当日值
        if prev_h is None:
            c_h_lag01 = val_h
            c_d_lag01 = val_d
        else:
            c_h_lag01 = 0.5 * (val_h + prev_h)
            c_d_lag01 = 0.5 * (val_d + prev_d)
        prev_h = val_h.copy()
        prev_d = val_d.copy()

        # 逐 ERF、逐阈值累加归因死亡人数
        for erf_name in ERF_SPECS:
            for thr_name, c0 in THRESHOLDS_UGM3.items():
                for b in ["mid", "low", "high"]:
                    beta = beta_table[erf_name][b]
                    if beta is None:
                        continue
                    m_h_grid = daily_base_deaths * calc_af(c_h_lag01, beta, c0)
                    m_d_grid = daily_base_deaths * calc_af(c_d_lag01, beta, c0)

                    tot_h = float(np.sum(m_h_grid))
                    tot_d = float(np.sum(m_d_grid))
                    acc_total[erf_name][thr_name][b]["annual_h"] += tot_h
                    acc_total[erf_name][thr_name][b]["annual_d"] += tot_d
                    if is_winter:
                        acc_total[erf_name][thr_name][b]["winter_h"] += tot_h
                        acc_total[erf_name][thr_name][b]["winter_d"] += tot_d

                    acc_monthly[erf_name][thr_name][month][f"h_{b}"] += tot_h
                    acc_monthly[erf_name][thr_name][month][f"d_{b}"] += tot_d

                    for pk in unique_provs:
                        pm = prov_masks[pk]
                        ph = float(np.sum(m_h_grid[pm]))
                        pd_val = float(np.sum(m_d_grid[pm]))
                        acc_prov[erf_name][thr_name][pk][f"h_{b}"] += ph
                        acc_prov[erf_name][thr_name][pk][f"d_{b}"] += pd_val
                        if is_winter and b == "mid":
                            acc_prov[erf_name][thr_name][pk]["win_h_mid"] += ph
                            acc_prov[erf_name][thr_name][pk]["win_d_mid"] += pd_val

        if (idx + 1) % 25 == 0 or (idx + 1) == len(grid_files):
            print(f"  -> 已完成 {idx + 1}/{len(grid_files)} 天短期归因死亡核算...")

        del gdf, daily_merged, X_hr, X_dy
        gc.collect()

    # =========================================================================
    # 4. 整理并导出 JSON / CSV 结果表
    # =========================================================================
    total_pop = float(np.sum(pop_aligned[pop_aligned > 0]))
    summary_out = {
        "Metadata": {
            "Study_Year": 2025,
            "Evaluated_Days": len(grid_files),
            "Total_Population": total_pop,
            "Total_Population_Millions": total_pop / 1e6,
            "Exposure_Window": "2-day moving average (lag 0-1)",
            "Baseline_Mortality_Source": "2025 Provincial Statistical Communiques (per 1,000 residents), National 8.04/1000 fallback",
            "Provincial_Rates_Per_1000": PROVINCE_MORTALITY_RATE_PER_1000,
        },
        "Results": {},
    }

    for erf_name, spec in ERF_SPECS.items():
        summary_out["Results"][erf_name] = {"Citation": spec["citation"], "Scenarios": {}}
        for thr_name, c0 in THRESHOLDS_UGM3.items():
            mid = acc_total[erf_name][thr_name]["mid"]
            low = acc_total[erf_name][thr_name]["low"]
            high = acc_total[erf_name][thr_name]["high"]
            has_ci = beta_table[erf_name]["low"] is not None and beta_table[erf_name]["high"] is not None

            ann_diff = mid["annual_h"] - mid["annual_d"]
            win_diff = mid["winter_h"] - mid["winter_d"]

            scen_dict = {
                "Counterfactual_C0_ugm3": c0,
                "Annual_Hourly_to_Daily_Deaths": mid["annual_h"],
                "Annual_Direct_Daily_Deaths": mid["annual_d"],
                "Annual_Diff_Hourly_minus_Daily": ann_diff,
                "Annual_Diff_pct": ann_diff / max(mid["annual_d"], 1e-9) * 100.0,
                "Annual_Hourly_per_100k": mid["annual_h"] / total_pop * 1e5,
                "Annual_Direct_per_100k": mid["annual_d"] / total_pop * 1e5,
                "Winter_Hourly_to_Daily_Deaths": mid["winter_h"],
                "Winter_Direct_Daily_Deaths": mid["winter_d"],
                "Winter_Diff_Hourly_minus_Daily": win_diff,
                "Winter_Diff_pct": win_diff / max(mid["winter_d"], 1e-9) * 100.0,
            }
            if has_ci:
                scen_dict["Annual_Hourly_95UI"] = [low["annual_h"], high["annual_h"]]
                scen_dict["Annual_Direct_95UI"] = [low["annual_d"], high["annual_d"]]
                scen_dict["Annual_Diff_95UI"] = [low["annual_h"] - low["annual_d"], high["annual_h"] - high["annual_d"]]
                scen_dict["Winter_Diff_95UI"] = [low["winter_h"] - low["winter_d"], high["winter_h"] - high["winter_d"]]
            summary_out["Results"][erf_name]["Scenarios"][thr_name] = scen_dict

    json_path = os.path.join(OUTPUT_DIR, "EP_Mortality_HIA_Summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)

    # 分省汇总 CSV
    prov_rows = []
    for erf_name in ERF_SPECS:
        has_ci = beta_table[erf_name]["low"] is not None and beta_table[erf_name]["high"] is not None
        for thr_name in THRESHOLDS_UGM3:
            for pk in unique_provs:
                pm = prov_masks[pk]
                p_pop = float(np.sum(pop_aligned[pm]))
                p_rate = PROVINCE_MORTALITY_RATE_PER_1000.get(pk, NATIONAL_FALLBACK_RATE_PER_1000)
                rec = acc_prov[erf_name][thr_name][pk]
                diff_mid = rec["h_mid"] - rec["d_mid"]
                prov_rows.append({
                    "ERF_Model": erf_name,
                    "Threshold": thr_name,
                    "Province_CN": pk,
                    "Province_EN": PROVINCE_EN_NAME.get(pk, pk),
                    "Population_Millions": p_pop / 1e6,
                    "Mortality_Rate_per_1000": p_rate,
                    "Annual_Hourly_Deaths": rec["h_mid"],
                    "Annual_Direct_Deaths": rec["d_mid"],
                    "Annual_Diff_Deaths": diff_mid,
                    "Annual_Diff_pct": diff_mid / max(rec["d_mid"], 1e-9) * 100.0,
                    "Annual_Diff_95UI_Low": (rec["h_low"] - rec["d_low"]) if has_ci else np.nan,
                    "Annual_Diff_95UI_High": (rec["h_high"] - rec["d_high"]) if has_ci else np.nan,
                    "Annual_Diff_per_100k": diff_mid / max(p_pop, 1.0) * 1e5,
                    "Winter_Hourly_Deaths": rec["win_h_mid"],
                    "Winter_Direct_Deaths": rec["win_d_mid"],
                    "Winter_Diff_Deaths": rec["win_h_mid"] - rec["win_d_mid"],
                })
    prov_df = pd.DataFrame(prov_rows)
    prov_csv = os.path.join(OUTPUT_DIR, "EP_Mortality_HIA_Provincial_Summary.csv")
    prov_df.to_csv(prov_csv, index=False, encoding="utf-8-sig")

    # 逐月时间序列 CSV
    month_rows = []
    for erf_name in ERF_SPECS:
        has_ci = beta_table[erf_name]["low"] is not None and beta_table[erf_name]["high"] is not None
        for thr_name in THRESHOLDS_UGM3:
            for m in range(1, 13):
                rec = acc_monthly[erf_name][thr_name][m]
                month_rows.append({
                    "ERF_Model": erf_name,
                    "Threshold": thr_name,
                    "Month": m,
                    "Hourly_Deaths": rec["h_mid"],
                    "Direct_Deaths": rec["d_mid"],
                    "Diff_Deaths": rec["h_mid"] - rec["d_mid"],
                    "Diff_95UI_Low": (rec["h_low"] - rec["d_low"]) if has_ci else np.nan,
                    "Diff_95UI_High": (rec["h_high"] - rec["d_high"]) if has_ci else np.nan,
                })
    month_df = pd.DataFrame(month_rows)
    month_csv = os.path.join(OUTPUT_DIR, "EP_Mortality_HIA_Monthly_TimeSeries.csv")
    month_df.to_csv(month_csv, index=False, encoding="utf-8-sig")

    # =========================================================================
    # 5. 绘制 §3.3 出版级双面板图 (主情景: Liu_2019_AllCause, WHO_15)
    # =========================================================================
    plot_erf = "Liu_2019_AllCause"
    plot_thr = "WHO_15"
    sub_m = month_df[(month_df["ERF_Model"] == plot_erf) & (month_df["Threshold"] == plot_thr)].sort_values("Month")
    sub_p = prov_df[
        (prov_df["ERF_Model"] == plot_erf)
        & (prov_df["Threshold"] == plot_thr)
        & (prov_df["Province_CN"] != "其他/海岸边缘")
    ].sort_values("Annual_Diff_Deaths", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

    # Panel (a): 逐月双路径可归因死亡人数与差值
    ax = axes[0]
    months = sub_m["Month"].values
    x = np.arange(len(months))
    w = 0.36
    ax.bar(x - w / 2, sub_m["Hourly_Deaths"], width=w, color="#1f77b4", label="Hourly-to-Daily Aggregation")
    ax.bar(x + w / 2, sub_m["Direct_Deaths"], width=w, color="#ff7f0e", label="Direct Daily Training")
    ax.set_xticks(x)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Short-Term Attributable Deaths (persons)", fontsize=11, weight="bold")
    ax.set_title("(a) Monthly Attributable Mortality (WHO 15 $\\mu$g/m$^3$)", fontsize=12, weight="bold", pad=8)
    ax.legend(frameon=True, fontsize=9.5, loc="upper center")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel (b): 分省年度死亡负担差异 (Hourly - Direct) 及 95% UI
    ax2 = axes[1]
    y_pos = np.arange(len(sub_p))
    diffs = sub_p["Annual_Diff_Deaths"].values
    err_low = diffs - sub_p["Annual_Diff_95UI_Low"].values
    err_high = sub_p["Annual_Diff_95UI_High"].values - diffs
    ax2.barh(y_pos, diffs, color="#d62728", alpha=0.85, xerr=[err_low, err_high], capsize=3, error_kw={"lw": 1.1})
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sub_p["Province_EN"].values, fontsize=10)
    ax2.set_xlabel("Annual Attributable Mortality Difference ($\\Delta M$, persons)", fontsize=11, weight="bold")
    ax2.set_title("(b) Provincial Mortality Burden Underestimation (Hourly − Direct)", fontsize=12, weight="bold", pad=8)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "Fig_EP_Mortality_Burden_Comparison.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

    print("\n" + "=" * 88)
    print("🎉 42 号脚本执行完成！输出文件：")
    print(f"  📊 JSON 汇总 : {json_path}")
    print(f"  📑 分省 CSV  : {prov_csv}")
    print(f"  📈 逐月 CSV  : {month_csv}")
    print(f"  🖼️ 出版级图件: {fig_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()