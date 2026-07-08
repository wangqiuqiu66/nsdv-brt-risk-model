"""Calculate and plot PDPs for the NSDV BRT model."""

from __future__ import annotations

import math
import os
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_ROOT = Path(os.environ.get("BRT_OUTPUT_DIR", PROJECT_ROOT / "output"))
HOST_OUTPUT_DIR = OUTPUT_ROOT / "host"
NSDV_OUTPUT_DIR = OUTPUT_ROOT / "NSDV"
PDP_OUTPUT_DIR = NSDV_OUTPUT_DIR / "pdp"

#files
RISK_DATA_PATH = INPUT_DIR / "risk_data_all.xlsx"
NSDV_LOC_PATH = INPUT_DIR / "nsdv_location.xlsx"
EXCLUDED_FEATURES = {"Horses_density"}

HOST_TICKS = [
    "Rhipicephalus appendiculatus",
    "Haemaphysalis intermedia",
    "Haemaphysalis longicornis",
]

FIXED_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.005,
    "subsample": 0.75,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": 1,
}

N_SIMULATIONS = 100
NEG_SAMPLE_RATIO = 5
PDP_RC_THRESHOLD = 0.05
GRID_RESOLUTION = 50
FIG_DPI = 600

FEATURE_LABELS = {
    "elevation": "Elevation (m)",
    "Bio1": "Annual mean temperature (deg C)",
    "Bio2": "Annual mean diurnal range (deg C)",
    "Bio3": "Isothermality",
    "Bio4": "Temperature seasonality (SD)",
    "Bio5": "Max temperature of warmest month (deg C)",
    "Bio6": "Min temperature of coldest month (deg C)",
    "Bio7": "Temperature annual range (deg C)",
    "Bio8": "Mean temperature of wettest quarter (deg C)",
    "Bio9": "Mean temperature of driest quarter (deg C)",
    "Bio10": "Mean temperature of warmest quarter (deg C)",
    "Bio11": "Mean temperature of coldest quarter (deg C)",
    "Bio12": "Annual precipitation (mm)",
    "Bio13": "Precipitation of wettest month (mm)",
    "Bio14": "Precipitation of driest month (mm)",
    "Bio15": "Precipitation seasonality (CV)",
    "Bio16": "Precipitation of wettest quarter (mm)",
    "Bio17": "Precipitation of driest quarter (mm)",
    "Bio18": "Precipitation of warmest quarter (mm)",
    "Bio19": "Precipitation of coldest quarter (mm)",
    "Sheep_density": "Mean number of sheep per square kilometer",
    "Goats_density": "Mean number of goats per square kilometer",
    "Cattle_density": "Mean number of cattle per square kilometer",
    "Buffaloes_density": "Mean number of buffaloes per square kilometer",
    "LandCoverType_10": "Percentage coverage of tree cover",
    "LandCoverType_20": "Percentage coverage of shrubland",
    "LandCoverType_30": "Percentage coverage of grassland",
    "LandCoverType_40": "Percentage coverage of cropland",
    "LandCoverType_50": "Percentage coverage of built-up",
    "LandCoverType_60": "Percentage coverage of bare or sparse vegetation",
    "LandCoverType_70": "Percentage coverage of snow and ice",
    "LandCoverType_80": "Percentage coverage of permanent water",
    "LandCoverType_90": "Percentage coverage of herbaceous wetland",
    "LandCoverType_95": "Percentage coverage of mangroves",
    "LandCoverType_100": "Percentage coverage of moss and lichen",
    "feat_prob_Rhipicephalus_appendiculatus": "Distribution probability of R. appendiculatus",
    "feat_prob_Haemaphysalis_intermedia": "Distribution probability of H. intermedia",
    "feat_prob_Haemaphysalis_longicornis": "Distribution probability of H. longicornis",
}


def safe_filename(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^\w\-]", "_", name)).strip("_")


def load_nsdv_locations() -> pd.DataFrame:
    nsdv_loc = pd.read_excel(NSDV_LOC_PATH, sheet_name="Sheet1")
    required = {"gaul2_code", "is"}
    missing = required.difference(nsdv_loc.columns)
    if missing:
        raise ValueError(f"{NSDV_LOC_PATH} is missing columns: {sorted(missing)}")
    return nsdv_loc.loc[nsdv_loc["is"].ne(0) | nsdv_loc["is"].isna()].copy()


def merge_tick_predictions(full_data: pd.DataFrame) -> pd.DataFrame:
    """Append tick-vector predicted probabilities to the NSDV risk table."""

    #tick data
    merged = full_data.copy()
    for tick in HOST_TICKS:
        pred_file = (
            HOST_OUTPUT_DIR
            / tick
            / "unified_result"
            / "prediction"
            / f"{tick}_prediction_summary.csv"
        )
        if not pred_file.exists():
            raise FileNotFoundError(pred_file)
        tick_df = pd.read_csv(pred_file)
        if not {"gaul2_code", "mean_prob"}.issubset(tick_df.columns):
            raise ValueError(f"{pred_file} must contain gaul2_code and mean_prob")

        #combine predicted tick probability with environmental risk factors
        safe_name = tick.replace(" ", "_").replace(".", "")
        col_name = f"feat_prob_{safe_name}"
        temp = tick_df[["gaul2_code", "mean_prob"]].rename(columns={"mean_prob": col_name})
        merged = merged.merge(temp, on="gaul2_code", how="left")
        merged[col_name] = merged[col_name].fillna(0)
    return merged


def read_feature_list() -> list[str]:
    """Use variables with mean relative contribution of at least 5% for PDP models."""

    #relative contribution
    rc_path = NSDV_OUTPUT_DIR / "rc" / "NSDV.csv"
    if not rc_path.exists():
        raise FileNotFoundError(f"No relative-contribution file found for NSDV: {rc_path}")

    rc_df = pd.read_csv(rc_path)
    required_cols = {"Key", "rc", "round"}
    if not required_cols.issubset(rc_df.columns):
        raise ValueError(f"{rc_path} must contain columns: {sorted(required_cols)}")

    rc_mean = rc_df.pivot_table(index="round", columns="Key", values="rc", fill_value=0).mean()
    features = [
        feature
        for feature in rc_mean[rc_mean >= PDP_RC_THRESHOLD].sort_values(ascending=False).index
        if feature not in EXCLUDED_FEATURES
    ]
    if not features:
        raise ValueError(f"No features with RC >= {PDP_RC_THRESHOLD:.0%} for NSDV")
    return features


def prepare_nsdv_data() -> pd.DataFrame:
    """Create the NSDV presence-background table with tick probability covariates."""

    #risk factors and NSDV data
    full_raw = pd.read_excel(RISK_DATA_PATH)
    nsdv_loc = load_nsdv_locations()
    pos_codes = set(nsdv_loc["gaul2_code"].dropna().unique())
    full_data = merge_tick_predictions(full_raw)
    full_data["status"] = full_data["gaul2_code"].isin(pos_codes).astype(int)
    return full_data


def extract_pdp_arrays(result) -> tuple[np.ndarray, np.ndarray]:
    """Extract grid values and PDP averages from sklearn output."""

    avg_vals = np.asarray(result["average"]).squeeze()
    if avg_vals.ndim > 1:
        avg_vals = avg_vals[-1]
    grid_vals = np.asarray(result["grid_values"][0], dtype=float)
    return grid_vals, np.asarray(avg_vals, dtype=float)


def calculate_nsdv_pdp(full_data: pd.DataFrame, feature_list: list[str]) -> None:
    """Refit NSDV BRT models repeatedly and save raw PDP values."""

    #output files
    PDP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    storage = {feature: [] for feature in feature_list}

    #combine case and background counties
    pos_data = full_data[full_data["status"] == 1].copy()
    neg_pool = full_data[full_data["status"] == 0].copy()
    n_neg_needed = min(len(neg_pool), len(pos_data) * NEG_SAMPLE_RATIO)

    print(f"[NSDV] positives={len(pos_data)}, features={len(feature_list)}")
    for simulation in range(1, N_SIMULATIONS + 1):
        #step 1, refit BRT model
        if simulation == 1 or simulation % 10 == 0:
            print(f"[NSDV] PDP simulation {simulation}/{N_SIMULATIONS}", flush=True)

        neg_sample = neg_pool.sample(n=n_neg_needed, random_state=simulation, replace=False)
        data = pd.concat([pos_data, neg_sample], ignore_index=True)

        train, _ = train_test_split(
            data,
            test_size=0.25,
            random_state=simulation,
            stratify=data["status"],
        )

        X = train[feature_list]
        y = train["status"]

        params = FIXED_PARAMS.copy()
        params["scale_pos_weight"] = (y == 0).sum() / max((y == 1).sum(), 1)
        model = xgb.XGBClassifier(random_state=simulation, **params)
        model.fit(X, y)

        #step 2, calculate PDP for each key risk factor
        for feature in feature_list:
            pdp_result = partial_dependence(
                model,
                X,
                features=[feature],
                response_method="predict_proba",
                grid_resolution=GRID_RESOLUTION,
                kind="average",
            )
            grid_vals, avg_vals = extract_pdp_arrays(pdp_result)
            for grid_val, avg_val in zip(grid_vals, avg_vals):
                storage[feature].append(
                    {
                        "simulation_id": simulation,
                        "feature_value": grid_val,
                        "pdp_value": avg_val,
                    }
                )

    #raw PDP data
    for feature in feature_list:
        pd.DataFrame(storage[feature]).to_csv(
            PDP_OUTPUT_DIR / f"NSDV_{safe_filename(feature)}.csv",
            index=False,
        )


def plot_nsdv_pdp(feature_list: list[str]) -> Path:
    """Plot the mean PDP curve and 95% interval for NSDV."""

    #PDP figure
    n_features = len(feature_list)
    n_cols = min(3, n_features)
    n_rows = math.ceil(n_features / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = np.ravel(axes)

    for idx, feature in enumerate(feature_list):
        #read raw PDP data
        ax = axes[idx]
        csv_path = PDP_OUTPUT_DIR / f"NSDV_{safe_filename(feature)}.csv"
        if not csv_path.exists():
            ax.text(0.5, 0.5, "Data Not Found", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        df = pd.read_csv(csv_path)
        x_min = df["feature_value"].min()
        x_max = df["feature_value"].max()
        x_interp = np.linspace(x_min, x_max, 100)

        interp_ys = []
        for simulation_id in df["simulation_id"].unique():
            sim_data = (
                df[df["simulation_id"] == simulation_id]
                .drop_duplicates("feature_value")
                .sort_values("feature_value")
            )
            if len(sim_data) >= 2:
                interp_ys.append(
                    np.interp(x_interp, sim_data["feature_value"], sim_data["pdp_value"])
                )
        if not interp_ys:
            ax.set_axis_off()
            continue

        #calculate mean and 95% CI
        interp_arr = np.asarray(interp_ys)
        mean_y = interp_arr.mean(axis=0)
        ci_low = np.percentile(interp_arr, 2.5, axis=0)
        ci_high = np.percentile(interp_arr, 97.5, axis=0)

        scale = 100 if "LandCoverType" in feature else 1
        ax.plot(x_interp * scale, mean_y, color="#d62728", linewidth=1.5)
        ax.fill_between(
            x_interp * scale,
            ci_low,
            ci_high,
            color="lightgrey",
            alpha=0.9,
            edgecolor="none",
        )
        label = FEATURE_LABELS.get(feature, feature)
        ax.set_xlabel(textwrap.fill(label, width=30), fontsize=18)
        ax.set_ylabel("Fitted function", fontsize=18)
        ax.tick_params(labelsize=16)
        ax.set_ylim(bottom=0)

    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    fig.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.5)
    save_path = PDP_OUTPUT_DIR / "NSDV_PDP_Combined.jpg"
    tif_path = PDP_OUTPUT_DIR / "NSDV_PDP_Combined.tif"
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(tif_path, dpi=FIG_DPI, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"[NSDV] saved {save_path}")
    print(f"[NSDV] saved {tif_path}")
    return save_path


def main() -> None:
    #figure style
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    #input files
    full_data = prepare_nsdv_data()
    feature_list = read_feature_list()

    #calculate and plot PDP
    calculate_nsdv_pdp(full_data, feature_list)
    plot_nsdv_pdp(feature_list)


if __name__ == "__main__":
    main()

