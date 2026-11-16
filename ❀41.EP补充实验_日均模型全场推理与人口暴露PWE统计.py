import os, glob, gc, json, warnings
import numpy as np
import pandas as pd
import xarray as xr
import joblib

warnings.filterwarnings("ignore")

BASE_DIR = "/home/wangzonghan/bisheshuju"
INFERENCE_DIR = "/data/Machine_Learning/wangzonghan/YRD_PM25_Project/Inference_Features"
POP_NC_PATH = f"{BASE_DIR}/Population/YRD_Population_1km_2025.nc"
HOURLY_MODEL_PATH = f"{BASE_DIR}/Results/Models_随机森林/best_rf_model_terrain.pkl"
HOURLY_FEAT_PATH = f"{BASE_DIR}/Results/Models_随机森林/best_features_list_terrain.pkl"
DAILY_MODEL_PATH = f"{BASE_DIR}/Results/Models_对照组_日均RF/best_rf_daily_model.pkl"
DAILY_FEAT_PATH = f"{BASE_DIR}/Results/Models_对照组_日均RF/best_features_list_daily.pkl"
OUTPUT_DIR = f"{BASE_DIR}/Results/EP_Supplementary_Stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def batch_predict(model, X, batch_size=250000):
    preds = []
    for i in range(0, X.shape[0], batch_size):
        bp = model.predict(X[i : i + batch_size])
        if hasattr(bp, "get"): bp = bp.get()
        elif hasattr(bp, "to_numpy"): bp = bp.to_numpy()
        preds.append(bp)
    return np.concatenate(preds)

def main():
    hr_model, hr_feats = joblib.load(HOURLY_MODEL_PATH), joblib.load(HOURLY_FEAT_PATH)
    dy_model, dy_feats = joblib.load(DAILY_MODEL_PATH), joblib.load(DAILY_FEAT_PATH)
    ds_pop = xr.open_dataset(POP_NC_PATH)
    pop_df = ds_pop.to_dataframe().reset_index().rename(columns={"longitude": "lon", "latitude": "lat"})
    pop_col = [c for c in pop_df.columns if c not in ["lon", "lat", "spatial_ref"]][0]
    pop_df = pop_df[["lon", "lat", pop_col]].rename(columns={pop_col: "pop"})
    pop_df["lon"], pop_df["lat"] = pop_df["lon"].round(3), pop_df["lat"].round(3)
    grid_files = sorted(glob.glob(f"{INFERENCE_DIR}/YRD_Grid_Hourly_Features_Terrain_2025*.parquet"))
    n_all, n_win = 0, 0
    sh_all, sd_all, sh_win, sd_win, grid_coords, pop_aligned = None, None, None, None, None, None
    exc_h, exc_d = 0.0, 0.0
    for fpath in grid_files:
        date_str = os.path.basename(fpath).split("_")[-1].split(".")[0]
        is_win = int(date_str[4:6]) in [12, 1, 2]
        gdf = pd.read_parquet(fpath)
        gdf["lon"], gdf["lat"] = gdf["lon"].round(3), gdf["lat"].round(3)
        gdf["pred_h"] = np.clip(batch_predict(hr_model, gdf[hr_feats].astype("float32").values), 0, 300)
        daily_h = gdf.groupby(["lat", "lon"], sort=True)["pred_h"].mean().reset_index()
        daily_f = gdf.groupby(["lat", "lon"], sort=True)[dy_feats].mean().reset_index()
        pred_d = np.clip(batch_predict(dy_model, daily_f[dy_feats].astype("float32").values), 0, 300)
        if grid_coords is None:
            grid_coords = daily_h[["lon", "lat"]].copy()
            mp = pd.merge(grid_coords, pop_df, on=["lon", "lat"], how="left")
            pop_aligned = np.maximum(np.nan_to_num(mp["pop"].values, nan=0.0), 0.0)
            sh_all, sd_all = np.zeros(len(grid_coords)), np.zeros(len(grid_coords))
            sh_win, sd_win = np.zeros(len(grid_coords)), np.zeros(len(grid_coords))
        vh, vd = daily_h["pred_h"].values, pred_d
        sh_all += vh; sd_all += vd; n_all += 1
        if is_win: sh_win += vh; sd_win += vd; n_win += 1
        exc_h += float(np.sum(pop_aligned[vh >= 75.0]))
        exc_d += float(np.sum(pop_aligned[vd >= 75.0]))
        del gdf, daily_h, daily_f; gc.collect()
    mh_ann, md_ann = sh_all / max(n_all, 1), sd_all / max(n_all, 1)
    mh_win, md_win = sh_win / max(n_win, 1), sd_win / max(n_win, 1)
    vm = pop_aligned > 0
    t_pop, n_land = float(np.sum(pop_aligned[vm])), int(np.sum(vm))
    pwe_ha, pwe_da = float(np.sum(pop_aligned * mh_ann) / t_pop), float(np.sum(pop_aligned * md_ann) / t_pop)
    pwe_hw, pwe_dw = float(np.sum(pop_aligned * mh_win) / t_pop), float(np.sum(pop_aligned * md_win) / t_pop)
    res = {
        "Population_Weighted_Exposure_PWE_ugm3": {"Annual_Hourly_to_Daily": pwe_ha, "Annual_Direct_Daily": pwe_da, "Annual_Diff": pwe_ha - pwe_da, "Winter_Hourly_to_Daily": pwe_hw, "Winter_Direct_Daily": pwe_dw, "Winter_Diff": pwe_hw - pwe_dw},
        "Exceedance_Person_Days_PM25_ge_75_Millions": {"Hourly_to_Daily": exc_h / 1e6, "Direct_Daily": exc_d / 1e6, "Diff": (exc_h - exc_d) / 1e6},
        "Exceeding_GB3095_Annual_Limit_35ugm3": {"Annual_Land_Area_pct": float(np.sum(mh_ann[vm] > 35.0) / n_land * 100.0), "Annual_Population_pct": float(np.sum(pop_aligned[mh_ann > 35.0]) / t_pop * 100.0), "Winter_Land_Area_pct": float(np.sum(mh_win[vm] > 35.0) / n_land * 100.0), "Winter_Population_pct": float(np.sum(pop_aligned[mh_win > 35.0]) / t_pop * 100.0)},
    }
    with open(os.path.join(OUTPUT_DIR, "EP_Grid_PWE_Exposure_Stats.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
