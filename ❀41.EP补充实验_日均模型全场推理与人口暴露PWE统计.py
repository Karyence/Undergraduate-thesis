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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

try:
    import cudf
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif", "Liberation Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

# =============================================================================
# 0. 基础路径配置（使用已保存的小时模型与直接日均模型进行全网格推理，不重新训练）
# =============================================================================
BASE_DIR = "/home/wangzonghan/bisheshuju"
INFERENCE_DIR = "/data/Machine_Learning/wangzonghan/YRD_PM25_Project/Inference_Features"
POP_NC_PATH = f"{BASE_DIR}/Population/YRD_Population_1km_2025.nc"
LC_NC_PATH = f"{BASE_DIR}/LandCover/YRD_LandCover_Fractions_1km_Final.nc"
TERRAIN_NC_PATH = f"{BASE_DIR}/GEBCO/GEBCO_2025_YRD_1km_with_slope.nc"

HOURLY_MODEL_PATH = f"{BASE_DIR}/Results/Models_随机森林/best_rf_model_terrain.pkl"
HOURLY_FEAT_PATH = f"{BASE_DIR}/Results/Models_随机森林/best_features_list_terrain.pkl"

DAILY_MODEL_PATH = f"{BASE_DIR}/Results/Models_对照组_日均RF/best_rf_daily_model.pkl"
DAILY_FEAT_PATH = f"{BASE_DIR}/Results/Models_对照组_日均RF/best_features_list_daily.pkl"

PROVINCE_SHP_PATH = "/home/yanchengzhu/Map/GIS/中国_省.shp"
CITY_SHP_PATH = "/home/yanchengzhu/Map/GIS/中国_市.shp"

OUTPUT_DIR = f"{BASE_DIR}/Results/EP_Supplementary_Stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    """
    由于 24.特征提取.py 生成网格 Parquet 时仅保存了小时模型筛选后的 24 个特征（剔除了 LC_shrubland_frac），
    而 35.对照实验_直接日均训练模型.py 的 best_features_list_daily.pkl 中保留了 LC_shrubland_frac，
    因此直接从 YRD_LandCover_Fractions_1km_Final.nc 中读取静态图层进行精确补齐。
    """
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


def main():
    print("=" * 85)
    print("🚀 启动 EP 补充统计：双路径 1km 全网格推理、人口加权暴露 (PWE) 与超标面积/人口核算")
    print("=" * 85)

    hr_model = joblib.load(HOURLY_MODEL_PATH)
    hr_feats = list(joblib.load(HOURLY_FEAT_PATH))
    dy_model = joblib.load(DAILY_MODEL_PATH)
    dy_feats = list(joblib.load(DAILY_FEAT_PATH))

    print(f"  -> 小时模型特征数: {len(hr_feats)} | 日均模型特征数: {len(dy_feats)}")

    # 加载 WorldPop 1km 网格人口数据
    ds_pop = xr.open_dataset(POP_NC_PATH)
    pop_df = ds_pop.to_dataframe().reset_index().rename(columns={"longitude": "lon", "latitude": "lat"})
    pop_col = [c for c in pop_df.columns if c not in ["lon", "lat", "spatial_ref"]][0]
    pop_df = pop_df[["lon", "lat", pop_col]].rename(columns={pop_col: "pop"})
    pop_df["lon"] = pop_df["lon"].round(3)
    pop_df["lat"] = pop_df["lat"].round(3)
    pop_df = pop_df.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)

    grid_files = sorted(glob.glob(f"{INFERENCE_DIR}/YRD_Grid_Hourly_Features_Terrain_2025*.parquet"))
    if not grid_files:
        print(f"❌ 未找到网格推理特征文件: {INFERENCE_DIR}")
        return

    # 预检查首个网格文件缺失的日均模型特征（如 LC_shrubland_frac）并预加载静态图层
    sample_cols = pd.read_parquet(grid_files[0], columns=["lon", "lat"]).columns.tolist()
    all_parquet_cols = pd.read_parquet(grid_files[0]).columns.tolist()
    missing_dy_cols = [c for c in dy_feats if c not in all_parquet_cols]
    present_dy_cols = [c for c in dy_feats if c in all_parquet_cols]

    static_lc_patch = None
    if missing_dy_cols:
        print(f"  🔧 检测到日均模型需要的特征 {missing_dy_cols} 不在日网格 Parquet 中，正在从静态土地覆盖 NC 文件补齐...")
        static_lc_patch = load_static_missing_lc_features(missing_dy_cols)

    n_days_total = 0
    n_days_winter = 0
    sum_hourly_agg_all = None
    sum_direct_daily_all = None
    sum_hourly_agg_win = None
    sum_direct_daily_win = None
    grid_coords = None
    aligned_missing_Arrays = {}

    exceed_pop_days_hourly_agg = 0.0
    exceed_pop_days_direct_daily = 0.0
    pop_aligned = None

    for idx, fpath in enumerate(grid_files):
        date_str = os.path.basename(fpath).split("_")[-1].split(".")[0]
        month = int(date_str[4:6])
        is_winter = month in [12, 1, 2]

        gdf = pd.read_parquet(fpath)
        gdf["lon"] = gdf["lon"].round(3)
        gdf["lat"] = gdf["lat"].round(3)

        # 1. 小时训练-日均聚合路径
        for f_col in hr_feats:
            if f_col not in gdf.columns:
                gdf[f_col] = 0.0
        X_hr = gdf[hr_feats].astype("float32").values
        gdf["pred_h"] = np.clip(batch_predict(hr_model, X_hr), 0, 300)

        # 使用 GPU (cuDF) 或 Pandas 聚合为日均网格（同时聚合 pred_h 与日均模型所需特征）
        agg_cols = ["pred_h"] + [c for c in present_dy_cols if c != "pred_h"]
        if HAS_CUDF:
            cgdf = cudf.from_pandas(gdf[["lat", "lon"] + agg_cols])
            daily_merged = (
                cgdf.groupby(["lat", "lon"], sort=True)[agg_cols]
                .mean()
                .reset_index()
                .to_pandas()
            )
            del cgdf
        else:
            daily_merged = (
                gdf.groupby(["lat", "lon"], sort=True)[agg_cols]
                .mean()
                .reset_index()
            )

        # 首日初始化空间坐标对齐、人口对齐与缺失静态特征（LC_shrubland_frac）对齐
        if grid_coords is None:
            grid_coords = daily_merged[["lon", "lat"]].copy()
            merged_pop = pd.merge(grid_coords, pop_df, on=["lon", "lat"], how="left")
            pop_aligned = np.nan_to_num(merged_pop["pop"].values, nan=0.0)
            pop_aligned = np.maximum(pop_aligned, 0.0)

            if static_lc_patch is not None:
                merged_lc = pd.merge(grid_coords, static_lc_patch, on=["lon", "lat"], how="left")
                for mc in missing_dy_cols:
                    if mc in merged_lc.columns:
                        aligned_missing_Arrays[mc] = np.nan_to_num(merged_lc[mc].values, nan=0.0).astype("float32")
                    else:
                        aligned_missing_Arrays[mc] = np.zeros(len(grid_coords), dtype="float32")
            else:
                for mc in missing_dy_cols:
                    aligned_missing_Arrays[mc] = np.zeros(len(grid_coords), dtype="float32")

            n_cells = len(grid_coords)
            sum_hourly_agg_all = np.zeros(n_cells, dtype=np.float64)
            sum_direct_daily_all = np.zeros(n_cells, dtype=np.float64)
            sum_hourly_agg_win = np.zeros(n_cells, dtype=np.float64)
            sum_direct_daily_win = np.zeros(n_cells, dtype=np.float64)

        # 补齐日均模型缺失列并严格按 dy_feats 顺序排列后推理
        for mc in missing_dy_cols:
            daily_merged[mc] = aligned_missing_Arrays[mc]

        X_dy = daily_merged[dy_feats].astype("float32").values
        pred_d = np.clip(batch_predict(dy_model, X_dy), 0, 300)

        val_h = daily_merged["pred_h"].values
        val_d = pred_d

        sum_hourly_agg_all += val_h
        sum_direct_daily_all += val_d
        n_days_total += 1

        if is_winter:
            sum_hourly_agg_win += val_h
            sum_direct_daily_win += val_d
            n_days_winter += 1

        # 累加当日 >=75 μg/m³ 的暴露人口·日
        exceed_pop_days_hourly_agg += float(np.sum(pop_aligned[val_h >= 75.0]))
        exceed_pop_days_direct_daily += float(np.sum(pop_aligned[val_d >= 75.0]))

        if (idx + 1) % 25 == 0 or (idx + 1) == len(grid_files):
            print(f"  -> 已完成 {idx + 1}/{len(grid_files)} 天网格双路径推理...")

        del gdf, daily_merged, X_hr, X_dy
        gc.collect()

    mean_h_annual = sum_hourly_agg_all / max(n_days_total, 1)
    mean_d_annual = sum_direct_daily_all / max(n_days_total, 1)
    mean_h_winter = sum_hourly_agg_win / max(n_days_winter, 1)
    mean_d_winter = sum_direct_daily_win / max(n_days_winter, 1)

    valid_pop_mask = pop_aligned > 0
    total_pop = float(np.sum(pop_aligned[valid_pop_mask]))
    n_land_cells = int(np.sum(valid_pop_mask))

    pwe_h_annual = float(np.sum(pop_aligned * mean_h_annual) / total_pop)
    pwe_d_annual = float(np.sum(pop_aligned * mean_d_annual) / total_pop)
    pwe_h_winter = float(np.sum(pop_aligned * mean_h_winter) / total_pop)
    pwe_d_winter = float(np.sum(pop_aligned * mean_d_winter) / total_pop)

    area_exc35_annual_pct = float(np.sum(mean_h_annual[valid_pop_mask] > 35.0) / n_land_cells * 100.0)
    pop_exc35_annual_pct = float(np.sum(pop_aligned[mean_h_annual > 35.0]) / total_pop * 100.0)
    area_exc35_winter_pct = float(np.sum(mean_h_winter[valid_pop_mask] > 35.0) / n_land_cells * 100.0)
    pop_exc35_winter_pct = float(np.sum(pop_aligned[mean_h_winter > 35.0]) / total_pop * 100.0)

    exposure_summary = {
        "Total_Evaluated_Days": n_days_total,
        "Winter_Evaluated_Days": n_days_winter,
        "Total_Population_Millions": total_pop / 1e6,
        "Annual_Mean_PM25_Land_ugm3": {
            "Hourly_to_Daily": float(np.mean(mean_h_annual[valid_pop_mask])),
            "Direct_Daily": float(np.mean(mean_d_annual[valid_pop_mask])),
            "Diff_Hourly_minus_Daily": float(np.mean((mean_h_annual - mean_d_annual)[valid_pop_mask])),
        },
        "Winter_Mean_PM25_Land_ugm3": {
            "Hourly_to_Daily": float(np.mean(mean_h_winter[valid_pop_mask])),
            "Direct_Daily": float(np.mean(mean_d_winter[valid_pop_mask])),
            "Diff_Hourly_minus_Daily": float(np.mean((mean_h_winter - mean_d_winter)[valid_pop_mask])),
        },
        "Population_Weighted_Exposure_PWE_ugm3": {
            "Annual_Hourly_to_Daily": pwe_h_annual,
            "Annual_Direct_Daily": pwe_d_annual,
            "Annual_Diff_Hourly_minus_Daily": pwe_h_annual - pwe_d_annual,
            "Annual_Diff_pct": (pwe_h_annual - pwe_d_annual) / pwe_d_annual * 100.0,
            "Winter_Hourly_to_Daily": pwe_h_winter,
            "Winter_Direct_Daily": pwe_d_winter,
            "Winter_Diff_Hourly_minus_Daily": pwe_h_winter - pwe_d_winter,
            "Winter_Diff_pct": (pwe_h_winter - pwe_d_winter) / pwe_d_winter * 100.0,
        },
        "Exceedance_Person_Days_PM25_ge_75_Millions": {
            "Hourly_to_Daily_Million_Person_Days": exceed_pop_days_hourly_agg / 1e6,
            "Direct_Daily_Million_Person_Days": exceed_pop_days_direct_daily / 1e6,
            "Diff_Million_Person_Days": (exceed_pop_days_hourly_agg - exceed_pop_days_direct_daily) / 1e6,
            "Diff_pct": (exceed_pop_days_hourly_agg - exceed_pop_days_direct_daily)
            / max(exceed_pop_days_direct_daily, 1.0)
            * 100.0,
        },
        "Exceeding_GB3095_Annual_Limit_35ugm3": {
            "Annual_Land_Area_Exceed_pct": area_exc35_annual_pct,
            "Annual_Population_Exceed_pct": pop_exc35_annual_pct,
            "Winter_Land_Area_Exceed_pct": area_exc35_winter_pct,
            "Winter_Population_Exceed_pct": pop_exc35_winter_pct,
        },
    }

    json_out = os.path.join(OUTPUT_DIR, "EP_Grid_PWE_Exposure_Stats.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(exposure_summary, f, indent=2, ensure_ascii=False)

    # 绘制两套产品年均与冬季均值空间差值图 (批注 19③)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300, subplot_kw={"projection": ccrs.PlateCarree()})
    diff_annual = mean_h_annual - mean_d_annual
    diff_winter = mean_h_winter - mean_d_winter

    for ax, diff_vals, title in zip(
        axes,
        [diff_annual, diff_winter],
        ["(a) Annual Mean Difference (Hourly-Agg − Direct-Daily)", "(b) Winter Mean Difference (Hourly-Agg − Direct-Daily)"],
    ):
        ax.set_extent([114.5, 123.0, 27.0, 35.5], crs=ccrs.PlateCarree())
        sc = ax.scatter(
            grid_coords["lon"],
            grid_coords["lat"],
            c=diff_vals,
            cmap="RdBu_r",
            vmin=-5.0,
            vmax=5.0,
            s=1.2,
            marker="s",
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        ax.add_feature(cfeature.NaturalEarthFeature("physical", "ocean", "10m", facecolor="white"), edgecolor="none", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="black", zorder=4)
        if os.path.exists(PROVINCE_SHP_PATH):
            try:
                prov = cfeature.ShapelyFeature(
                    shpreader.Reader(PROVINCE_SHP_PATH).geometries(),
                    ccrs.PlateCarree(),
                    facecolor="none",
                    edgecolor="#222222",
                    linewidth=0.9,
                )
                ax.add_feature(prov, zorder=3)
            except Exception:
                pass
        # 1. 添加与论文其它地图一致的经纬度网格与加粗坐标刻度
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.4,
            linestyle="--",
            zorder=5,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 12, "weight": "bold", "family": "serif"}
        gl.ylabel_style = {"size": 12, "weight": "bold", "family": "serif"}

        # 2. 标题字号与间距微调（与正文其它双子图对齐）
        ax.set_title(title, fontsize=15, weight="bold", pad=12)

        # 3. Colorbar 刻度加粗 + 底部标签全加粗
        cb = plt.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.046, pad=0.07)
        cb.ax.tick_params(labelsize=12)
        for l in cb.ax.xaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_family("serif")
        cb.set_label(
            r"$\boldsymbol{\Delta}\mathbf{PM}_{\mathbf{2.5}}$ ($\boldsymbol{\mu}\mathbf{g}/\mathbf{m}^{\mathbf{3}}$)",
            fontsize=14,
            weight="bold",
            labelpad=6,
        )

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "Fig_EP_Annual_Winter_Pathway_Difference.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

    print("\n" + "=" * 85)
    print(f"🎉 全网格暴露统计与差值图生成完毕！\n📊 JSON: {json_out}\n🖼️ 差值图: {fig_path}")
    print("=" * 85)
    print(json.dumps(exposure_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()