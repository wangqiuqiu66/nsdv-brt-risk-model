"""Calculate and plot PDPs for tick-vector BRT models."""

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

#files
RISK_DATA_PATH = INPUT_DIR / "risk_data_all.xlsx"
TICK_LOC_PATH = INPUT_DIR / "tick_location.xlsx"

TARGET_SPECIES = [
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
}


def safe_filename(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^\w\-]", "_", name)).strip("_")


def read_feature_list(species_name: str) -> list[str]:
    """Use variables with relative contribution of at least 5% for PDP models."""

    #relative contribution
    result_dir = HOST_OUTPUT_DIR / species_name / "unified_result"
    rc_path = result_dir / "rc" / f"{species_name}_rc_aggregated.csv"

    rc_df = pd.read_csv(rc_path)
    features = (
        rc_df.loc[rc_df["rc"] >= PDP_RC_THRESHOLD]
        .sort_values("rc", ascending=False)["Key"]
        .tolist()
    )
    return features


def prepare_tick_data(
    species_name: str,
    full_risk_data: pd.DataFrame,
    tick_loc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a presence-background table for one tick species."""

    #tick data
    pos_codes = set(
        tick_loc_df.loc[tick_loc_df["Species"].eq(species_name), "gaul2_code"]
        .dropna()
        .unique()
    )
    data = full_risk_data.copy()
    data["status"] = data["gaul2_code"].isin(pos_codes).astype(int)
    return data


def extract_pdp_arrays(result) -> tuple[np.ndarray, np.ndarray]:
    """Extract grid values and PDP averages from sklearn output."""

    avg_vals = np.asarray(result["average"]).squeeze()
    if avg_vals.ndim > 1:
        avg_vals = avg_vals[-1]
    grid_vals = np.asarray(result["grid_values"][0], dtype=float)
    return grid_vals, np.asarray(avg_vals, dtype=float)


def calculate_species_pdp(
    species_name: str,
    feature_list: list[str],
    full_risk_data: pd.DataFrame,
    tick_loc_df: pd.DataFrame,
) -> None:
    """Refit BRT models repeatedly and save raw PDP values for each variable."""

    #output files
    pdp_dir = HOST_OUTPUT_DIR / species_name / "unified_result" / "pdp_data"
    pdp_dir.mkdir(parents=True, exist_ok=True)

    #combine case and background counties
    data = prepare_tick_data(species_name, full_risk_data, tick_loc_df)
    pos_data = data[data["status"] == 1].copy()
    neg_pool = data[data["status"] == 0].copy()
    n_neg_needed = min(len(neg_pool), len(pos_data) * NEG_SAMPLE_RATIO)
    storage = {feature: [] for feature in feature_list}

    print(f"[{species_name}] positives={len(pos_data)}, features={len(feature_list)}")
    for simulation in range(1, N_SIMULATIONS + 1):
        #step 1, refit BRT model
        if simulation == 1 or simulation % 10 == 0:
            print(f"[{species_name}] PDP simulation {simulation}/{N_SIMULATIONS}", flush=True)

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
    safe_species = species_name.replace(" ", "_")
    for feature in feature_list:
        pd.DataFrame(storage[feature]).to_csv(
            pdp_dir / f"{safe_species}_{safe_filename(feature)}.csv",
            index=False,
        )


def plot_species_pdp(species_name: str, feature_list: list[str]) -> Path:
    """Plot the mean PDP curve and 95% interval for one tick species."""

    #output files
    safe_species = species_name.replace(" ", "_")
    pdp_dir = HOST_OUTPUT_DIR / species_name / "unified_result" / "pdp_data"
    fig_dir = HOST_OUTPUT_DIR / species_name / "unified_result" / "pdp"
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_features = len(feature_list)
    n_cols = min(3, n_features)
    n_rows = math.ceil(n_features / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4))
    axes = np.ravel(axes)

    for idx, feature in enumerate(feature_list):
        #read raw PDP data
        ax = axes[idx]
        csv_path = pdp_dir / f"{safe_species}_{safe_filename(feature)}.csv"
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
    #PDP figure
    save_path = fig_dir / f"{safe_species}_PDP.jpg"
    tif_path = fig_dir / f"{safe_species}_PDP.tif"
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(tif_path, dpi=FIG_DPI, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"[{species_name}] saved {save_path}")
    print(f"[{species_name}] saved {tif_path}")
    return save_path


def main() -> None:
    #figure style
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    #input files
    full_risk_data = pd.read_excel(RISK_DATA_PATH)
    tick_loc_df = pd.read_excel(TICK_LOC_PATH)

    #run each tick species
    for species_name in TARGET_SPECIES:
        feature_list = read_feature_list(species_name)
        calculate_species_pdp(species_name, feature_list, full_risk_data, tick_loc_df)
        plot_species_pdp(species_name, feature_list)


if __name__ == "__main__":
    main()

