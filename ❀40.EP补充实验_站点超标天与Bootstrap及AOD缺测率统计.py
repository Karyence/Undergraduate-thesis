import os, glob, gc, json, warnings
import numpy as np
import pandas as pd
import xarray as xr
import rasterio, joblib
import chinese_calendar as conc
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = "/home/wangzonghan/bisheshuju"
DATA_FILE = f"{BASE_DIR}/训练集/YRD_PM25_Hourly_ML_Dataset_2025.parquet"
HOURLY_MODEL_PATH = f"{BASE_DIR}/Results/Models_随机森林/best_rf_model_terrain.pkl"
HOURLY_FEAT_PATH = f"{BASE_DIR}/Results/Models_随机森林/best_features_list_terrain.pkl"
DAILY_MODEL_PATH = f"{BASE_DIR}/Results/Models_对照组_日均RF/best_rf_daily_model.pkl"
DAILY_FEAT_PATH = f"{BASE_DIR}/Results/Models_对照组_日均RF/best_features_list_daily.pkl"
DAILY_PRED_CSV = f"{BASE_DIR}/Results/Models_对照组_日均RF/Daily_Model_Test_Results_14808.csv"
AOD_RAW_H5_DIR = "/data/Satellite/AOD_YRD"
TARGET_GRID_NC = f"{BASE_DIR}/GEBCO/GEBCO_2025_YRD_1km.nc"
OUTPUT_DIR = f"{BASE_DIR}/Results/EP_Supplementary_Stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def batch_predict(model, X, batch_size=25000):
    preds = []
    for i in range(0, X.shape[0], batch_size):
        bp = model.predict(X[i : i + batch_size])
        if hasattr(bp, "get"): bp = bp.get()
        elif hasattr(bp, "to_numpy"): bp = bp.to_numpy()
        preds.extend(bp)
    return np.array(preds, dtype=np.float64)

def calc_all_metrics(obs, pred):
    obs, pred = np.asarray(obs, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    return {
        "R2": float(r2_score(obs, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(obs, pred))),
        "MAE": float(mean_absolute_error(obs, pred)),
        "ME": float(np.mean(pred - obs)),
        "NMB": float(np.sum(pred - obs) / np.sum(obs) * 100.0),
        "NME": float(np.sum(np.abs(pred - obs)) / np.sum(obs) * 100.0),
    }

def calc_exceedance_contingency(obs, pred, threshold=75.0):
    obs_exc, pred_exc = obs >= threshold, pred >= threshold
    tp = int(np.sum(obs_exc & pred_exc))
    fn = int(np.sum(obs_exc & (~pred_exc)))
    fp = int(np.sum((~obs_exc) & pred_exc))
    tn = int(np.sum((~obs_exc) & (~pred_exc)))
    return {
        "Obs_Exceed_Days": int(tp + fn), "TP_Hits": tp, "FN_Misses": fn,
        "FP_FalseAlarms": fp, "TN_CorrectNegatives": tn,
        "POD_pct": tp / (tp + fn) * 100.0 if (tp + fn) > 0 else np.nan,
        "FAR_pct": fp / (tp + fp) * 100.0 if (tp + fp) > 0 else np.nan,
        "CSI_pct": tp / (tp + fn + fp) * 100.0 if (tp + fn + fp) > 0 else np.nan,
        "F1_pct": 2 * tp / (2 * tp + fp + fn) * 100.0 if (2 * tp + fp + fn) > 0 else np.nan,
    }

def run_station_block_bootstrap(df_daily, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    stations = df_daily["site_code"].unique()
    station_groups = {s: df_daily[df_daily["site_code"] == s] for s in stations}
    boot_records = []
    for _ in range(n_boot):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        bdf = pd.concat([station_groups[s] for s in sampled], ignore_index=True)
        obs, ph, pd_ = bdf["obs_daily"].values, bdf["pred_hourly_agg"].values, bdf["pred_direct_daily"].values
        mh, md = calc_all_metrics(obs, ph), calc_all_metrics(obs, pd_)
        hm = obs >= 75.0
        rh_h = np.sqrt(mean_squared_error(obs[hm], ph[hm])) if np.sum(hm) > 5 else np.nan
        rd_h = np.sqrt(mean_squared_error(obs[hm], pd_[hm])) if np.sum(hm) > 5 else np.nan
        boot_records.append({
            "R2_hourly": mh["R2"], "R2_daily": md["R2"], "dR2": mh["R2"] - md["R2"],
            "RMSE_hourly": mh["RMSE"], "RMSE_daily": md["RMSE"], "dRMSE": md["RMSE"] - mh["RMSE"],
            "MAE_hourly": mh["MAE"], "MAE_daily": md["MAE"], "dMAE": md["MAE"] - mh["MAE"],
            "RMSE_high_hourly": rh_h, "RMSE_high_daily": rd_h, "dRMSE_high": rd_h - rh_h,
        })
    boot_res = pd.DataFrame(boot_records)
    ci = {c: {"mean": float(boot_res[c].mean()), "ci95_low": float( np.percentile(boot_res[c].dropna(), 2.5)), "ci95_high": float(np.percentile(boot_res[c].dropna(), 97.5))} for c in boot_res.columns}
    sh = np.array([np.sqrt(mean_squared_error(station_groups[s]["obs_daily"], station_groups[s]["pred_hourly_agg"])) for s in stations])
    sd = np.array([np.sqrt(mean_squared_error(station_groups[s]["obs_daily"], station_groups[s]["pred_direct_daily"])) for s in stations])
    t_stat, p_t = stats.ttest_rel(sd, sh)
    w_stat, p_w = stats.wilcoxon(sd, sh)
    ci["paired_tests_44_stations"] = {"paired_t_stat": float(t_stat), "paired_t_pvalue": float(p_t), "wilcoxon_stat": float(w_stat), "wilcoxon_pvalue": float(p_w), "stations_improved_count": int(np.sum(sh < sd)), "stations_total": len(stations)}
    return ci

def main():
    df_hourly = pd.read_parquet(DATA_FILE)
    df_hourly["date"] = pd.to_datetime(df_hourly["date"])
    if "month" not in df_hourly.columns:
        df_hourly["month"] = df_hourly["date"].dt.month
        df_hourly["day_of_week"] = df_hourly["date"].dt.dayofweek
        df_hourly["is_weekend"] = df_hourly["day_of_week"].isin([5, 6]).astype(int)
        df_hourly["season"] = (df_hourly["month"] % 12 // 3 + 1).astype(int)
        df_hourly["is_holiday"] = df_hourly["date"].apply(lambda x: 1 if conc.is_holiday(x) else 0)
    sites_info = df_hourly[["site_code", "lon", "lat"]].drop_duplicates().reset_index(drop=True)
    kmeans = KMeans(n_clusters=int(len(sites_info) * 0.15), random_state=42)
    sites_info["spatial_cluster"] = kmeans.fit_predict(sites_info[["lon", "lat"]])
    test_sites = sites_info.groupby("spatial_cluster").apply(lambda x: x.sample(n=1, random_state=42)).reset_index(drop=True)["site_code"].tolist()
    test_df_hourly = df_hourly[df_hourly["site_code"].isin(test_sites)].copy()
    hr_model, hr_feats = joblib.load(HOURLY_MODEL_PATH), joblib.load(HOURLY_FEAT_PATH)
    test_df_hourly["pred_hourly"] = batch_predict(hr_model, test_df_hourly[hr_feats].astype("float32").values)
    del hr_model; gc.collect()
    daily_agg = test_df_hourly.groupby(["date", "site_code", "lon", "lat"]).agg(obs_daily=("pm25_hourly", "mean"), pred_hourly_agg=("pred_hourly", "mean"), valid_hours=("pm25_hourly", "count")).reset_index()
    df_dir_csv = pd.read_csv(DAILY_PRED_CSV)
    df_dir_csv["date"] = pd.to_datetime(df_dir_csv["date"])
    daily_agg = pd.merge(daily_agg, df_dir_csv[["date", "site_code", "predicted_pm25"]].rename(columns={"predicted_pm25": "pred_direct_daily"}), on=["date", "site_code"], how="inner")
    df_ge20 = daily_agg[daily_agg["valid_hours"] >= 20].copy()
    bins, labels = [0, 35, 75, 10000], ["Low (<35)", "Moderate (35-75)", "High (>=75)"]
    daily_agg["conc_group"] = pd.cut(daily_agg["obs_daily"], bins=bins, labels=labels, right=False)
    sf_start, sf_end = pd.Timestamp("2025-01-28"), pd.Timestamp("2025-02-04")
    sf_daily = daily_agg[(daily_agg["date"] >= sf_start) & (daily_agg["date"] <= sf_end)]
    win_other = daily_agg[(daily_agg["date"].dt.month.isin([12, 1, 2])) & ~((daily_agg["date"] >= sf_start) & (daily_agg["date"] <= sf_end))]
    report = {
        "all_station_days_14808": {"N": len(daily_agg), "Hourly_to_Daily": calc_all_metrics(daily_agg["obs_daily"], daily_agg["pred_hourly_agg"]), "Direct_Daily": calc_all_metrics(daily_agg["obs_daily"], daily_agg["pred_direct_daily"])},
        "ge20h_sensitivity": {"N_ge20h": len(df_ge20), "Retained_pct": float(len(df_ge20) / len(daily_agg) * 100.0), "Hourly_to_Daily": calc_all_metrics(df_ge20["obs_daily"], df_ge20["pred_hourly_agg"]), "Direct_Daily": calc_all_metrics(df_ge20["obs_daily"], df_ge20["pred_direct_daily"])},
        "bootstrap_44_stations": run_station_block_bootstrap(daily_agg),
        "exceedance_75ugm3": {"Hourly_to_Daily": calc_exceedance_contingency(daily_agg["obs_daily"].values, daily_agg["pred_hourly_agg"].values), "Direct_Daily": calc_exceedance_contingency(daily_agg["obs_daily"].values, daily_agg["pred_direct_daily"].values)},
        "concentration_stratified_metrics": {g: {"n": len(daily_agg[daily_agg["conc_group"] == g]), "Hourly_to_Daily": calc_all_metrics(daily_agg[daily_agg["conc_group"] == g]["obs_daily"], daily_agg[daily_agg["conc_group"] == g]["pred_hourly_agg"]), "Direct_Daily": calc_all_metrics(daily_agg[daily_agg["conc_group"] == g]["obs_daily"], daily_agg[daily_agg["conc_group"] == g]["pred_direct_daily"])} for g in labels},
        "spring_festival_all_test_sites": {"Spring_Festival_Jan28_Feb04": {"n": len(sf_daily), "Hourly_to_Daily": calc_all_metrics(sf_daily["obs_daily"], sf_daily["pred_hourly_agg"]), "Direct_Daily": calc_all_metrics(sf_daily["obs_daily"], sf_daily["pred_direct_daily"])}, "Other_Winter_Days": {"n": len(win_other), "Hourly_to_Daily": calc_all_metrics(win_other["obs_daily"], win_other["pred_hourly_agg"]), "Direct_Daily": calc_all_metrics(win_other["obs_daily"], win_other["pred_direct_daily"])}},
    }
    with open(os.path.join(OUTPUT_DIR, "EP_Station_Supplementary_Stats.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
