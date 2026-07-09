"""Run the NSDV BRT model.

The NSDV model uses environmental variables and the predicted occurrence
probabilities of the three tick vectors as risk factors. Tick models must be
run before this script.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, Side
from openpyxl.utils import get_column_letter
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

# ---- inline helpers (formerly brt_utils) ----

def _feature_importance_frame(model) -> pd.DataFrame:
    importance = model.get_booster().get_score(importance_type="weight")
    frame = pd.DataFrame(list(importance.items()), columns=["Key", "Value"])
    frame["rc"] = frame["Value"] / frame["Value"].sum()
    return frame.sort_values("rc", ascending=False).reset_index(drop=True)


def _choose_key_factors(importance_df: pd.DataFrame, min_rc: float) -> list[str]:
    return importance_df.loc[importance_df["rc"] >= min_rc, "Key"].tolist()


def _summarize_metrics(metrics_df: pd.DataFrame, label: str) -> pd.DataFrame:
    stats = {"species": label}
    for metric in ["auc", "tpr", "tnr", "threshold"]:
        if metric in metrics_df.columns and not metrics_df[metric].empty:
            values = metrics_df[metric]
            stats[f"{metric}_mean"] = values.mean()
            stats[f"{metric}_lbd"] = np.percentile(values, 2.5)
            stats[f"{metric}_ubd"] = np.percentile(values, 97.5)
    return pd.DataFrame([stats]).round(3)


def _summarize_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    return (
        pred_df.groupby("gaul2_code")["pred"]
        .agg(
            mean_prob="mean",
            ci_lower=lambda x: np.percentile(x, 2.5),
            ci_upper=lambda x: np.percentile(x, 97.5),
        )
        .reset_index()
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_ROOT = Path(os.environ.get("BRT_OUTPUT_DIR", PROJECT_ROOT / "output"))
OUTPUT_BASE = OUTPUT_ROOT / "NSDV"
HOST_OUTPUT_DIR = OUTPUT_ROOT / "host"
RC_TABLE_DIR = OUTPUT_ROOT / "nsdv_rc_tables"

NON_ENVIRONMENT_COLUMNS = {
    "gaul2_code", "gaul0_name", "gaul1_name", "gaul2_name", "continent",
    "status", "model_weight", "rescaled", "prob", "recip",
}

#files
RISK_DATA_PATH = INPUT_DIR / "risk_data_all.xlsx"
NSDV_LOC_PATH = INPUT_DIR / "nsdv_location.xlsx"
WEIGHT_PATH = INPUT_DIR / "weight.xlsx"

HOST_TICKS = [
    "Rhipicephalus appendiculatus",
    "Haemaphysalis intermedia",
    "Haemaphysalis longicornis",
]

SCREEN_ITERATIONS = 10
ITERATIONS = 100
NEG_SAMPLE_RATIO = 5
KEY_FACTOR_MIN_RC = 0.02
HIGH_RISK_THRESHOLD = 0.5

TICK_PROB_FEATURES = [
    "feat_prob_Rhipicephalus_appendiculatus",
    "feat_prob_Haemaphysalis_intermedia",
    "feat_prob_Haemaphysalis_longicornis",
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

RISK_FACTOR_LABELS = {
    "elevation": "Elevation",
    "Bio1": "Bio1", "Bio2": "Bio2", "Bio3": "Bio3", "Bio4": "Bio4",
    "Bio5": "Bio5", "Bio6": "Bio6", "Bio7": "Bio7", "Bio8": "Bio8",
    "Bio9": "Bio9", "Bio10": "Bio10", "Bio11": "Bio11", "Bio12": "Bio12",
    "Bio13": "Bio13", "Bio14": "Bio14", "Bio15": "Bio15", "Bio16": "Bio16",
    "Bio17": "Bio17", "Bio18": "Bio18", "Bio19": "Bio19",
    "LandCoverType_10": "Tree cover",
    "LandCoverType_20": "Shrubland", "LandCoverType_30": "Grassland",
    "LandCoverType_40": "Cropland", "LandCoverType_50": "Built-up",
    "LandCoverType_60": "Bare/sparse vegetation", "LandCoverType_70": "Snow and ice",
    "LandCoverType_80": "Permanent water bodies", "LandCoverType_90": "Herbaceous wetland",
    "LandCoverType_95": "Mangroves", "LandCoverType_100": "Moss and lichen",
    "Buffaloes_density": "Buffaloes", "Sheep_density": "Sheep",
    "Goats_density": "Goats", "Cattle_density": "Cattle",
    "feat_prob_Rhipicephalus_appendiculatus": "Predicted probability of R. appendiculatus",
    "feat_prob_Haemaphysalis_intermedia": "Predicted probability of H. intermedia",
    "feat_prob_Haemaphysalis_longicornis": "Predicted probability of H. longicornis",
}


def get_environment_features(df: pd.DataFrame) -> list[str]:
    return [
        col for col in df.columns
        if col not in NON_ENVIRONMENT_COLUMNS
        and col not in TICK_PROB_FEATURES
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def load_nsdv_locations() -> pd.DataFrame:
    nsdv_loc = pd.read_excel(NSDV_LOC_PATH, sheet_name="Sheet1")
    required = {"gaul2_code"}
    missing = required.difference(nsdv_loc.columns)
    if missing:
        raise ValueError(f"{NSDV_LOC_PATH} is missing columns: {sorted(missing)}")
    return nsdv_loc.copy()


def add_model_weights(full_data: pd.DataFrame) -> pd.DataFrame:
    """Merge cities survey weights and use 1 for cities without a weight."""

    weight_df = pd.read_excel(WEIGHT_PATH)

    weighted = full_data.copy()
    weighted["gaul2_code"] = weighted["gaul2_code"].astype(str)
    weight_df = weight_df[["gaul2_code", "rescaled"]].copy()
    weight_df["gaul2_code"] = weight_df["gaul2_code"].astype(str)

    weighted = weighted.merge(weight_df, on="gaul2_code", how="left")
    weighted["model_weight"] = weighted["rescaled"].fillna(1.0)
    return weighted.drop(columns=["rescaled"])


def merge_tick_predictions(full_data: pd.DataFrame) -> pd.DataFrame:
    """Append tick-vector predicted probabilities to the risk-factor table."""

    #tick data
    merged_data = full_data.copy()
    for tick in HOST_TICKS:
        pred_file = (
            HOST_OUTPUT_DIR
            / tick
            / "unified_result"
            / "prediction"
            / f"{tick}_prediction_summary.csv"
        )
        tick_pred_df = pd.read_csv(pred_file)

        #combine predicted tick probability with environmental risk factors
        safe_name = tick.replace(" ", "_").replace(".", "")
        col_name = f"feat_prob_{safe_name}"
        temp_df = tick_pred_df[["gaul2_code", "mean_prob"]].rename(columns={"mean_prob": col_name})
        merged_data = merged_data.merge(temp_df, on="gaul2_code", how="left")
        merged_data[col_name] = merged_data[col_name].fillna(0)

    return merged_data


def sample_case_control(
    full_data: pd.DataFrame,
    nsdv_pos_codes: set,
    simulation: int,
) -> pd.DataFrame:
    """Build one NSDV case-control sample at a 1:5 case-to-control ratio."""

    random.seed(simulation)
    #case counties
    case_df = full_data[full_data["gaul2_code"].isin(nsdv_pos_codes)].copy()
    case_df["status"] = 1

    #control counties
    all_codes = set(full_data["gaul2_code"].unique())
    potential_control_codes = sorted(all_codes - set(case_df["gaul2_code"].unique()))
    n_neg_needed = len(case_df) * NEG_SAMPLE_RATIO

    selected_control_codes = random.sample(potential_control_codes, n_neg_needed)

    control_df = full_data[full_data["gaul2_code"].isin(selected_control_codes)].copy()
    control_df["status"] = 0
    return pd.concat([case_df, control_df], ignore_index=True)


def screen_key_factors(
    full_data: pd.DataFrame,
    nsdv_pos_codes: set,
    feature_cols: list[str],
) -> tuple[list[str], pd.DataFrame]:
    """Identify key NSDV risk factors from repeated screening models."""

    #step 1, identify key risk factors
    importance_frames = []

    for simulation in range(SCREEN_ITERATIONS):
        #combine and split
        survey_data = sample_case_control(full_data, nsdv_pos_codes, simulation)

        train, _ = train_test_split(
            survey_data,
            test_size=0.25,
            random_state=simulation,
            stratify=survey_data["status"],
        )

        n_train_neg = (train["status"] == 0).sum()
        n_train_pos = (train["status"] == 1).sum()

        params = FIXED_PARAMS.copy()
        params["scale_pos_weight"] = n_train_neg / n_train_pos

        model = xgb.XGBClassifier(random_state=simulation, **params)
        model.fit(
            train[feature_cols],
            train["status"],
            sample_weight=train["model_weight"],
        )

        #relative contribution
        importance = _feature_importance_frame(model)
        importance["round"] = simulation
        importance_frames.append(importance)

    #combine relative contribution from all screening rounds
    rc_df = pd.concat(importance_frames, ignore_index=True)
    if rc_df.empty or rc_df["Value"].sum() == 0:
        rc_grouped = pd.DataFrame(columns=["Key", "sum_value", "rc"])
        return [], rc_grouped

    rc_grouped = rc_df.groupby("Key")["Value"].sum().reset_index(name="sum_value")
    rc_grouped["rc"] = rc_grouped["sum_value"] / rc_grouped["sum_value"].sum()
    key_factors = _choose_key_factors(rc_grouped, KEY_FACTOR_MIN_RC)
    return key_factors, rc_grouped.sort_values("rc", ascending=False).reset_index(drop=True)


def fit_one_nsdv_model(
    full_data: pd.DataFrame,
    nsdv_pos_codes: set,
    key_factors: list[str],
    simulation: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one bootstrap NSDV model and return prediction, metric, HRC, and RC tables."""

    #step 2, predict use key risk factors
    random.seed(simulation)
    np.random.seed(simulation)

    #combine and split
    survey_data = sample_case_control(full_data, nsdv_pos_codes, simulation)

    train, test = train_test_split(
        survey_data,
        test_size=0.25,
        random_state=simulation,
        stratify=survey_data["status"],
    )

    n_train_neg = (train["status"] == 0).sum()
    n_train_pos = (train["status"] == 1).sum()
    current_params = FIXED_PARAMS.copy()
    current_params["scale_pos_weight"] = n_train_neg / n_train_pos

    model = xgb.XGBClassifier(random_state=simulation, **current_params)
    model.fit(
        train[key_factors],
        train["status"],
        sample_weight=train["model_weight"],
    )

    #relative contribution
    rel_con_df = _feature_importance_frame(model)
    rel_con_df["round"] = simulation

    #model evaluation by testing set
    prob_test = model.predict_proba(test[key_factors])[:, 1]
    fpr, tpr, thresholds = roc_curve(test["status"], prob_test)
    optimal_idx = int(np.argmax(tpr - fpr))
    threshold = thresholds[optimal_idx]

    metrics_df = pd.DataFrame(
        {
            "round": [simulation],
            "auc": [auc(fpr, tpr)],
            "tpr": [tpr[optimal_idx]],
            "tnr": [1 - fpr[optimal_idx]],
            "threshold": [threshold],
            "n_features_final": [len(key_factors)],
        }
    )

    #prediction probability for all counties
    pred_prob = model.predict_proba(full_data[key_factors])[:, 1]
    pred_df = pd.DataFrame(
        {
            "gaul2_code": full_data["gaul2_code"],
            "round": simulation,
            "pred": pred_prob,
        }
    )

    #high_risk_counties
    high_risk_df = pred_df[pred_df["pred"] >= HIGH_RISK_THRESHOLD].copy()
    high_risk_df["threshold"] = HIGH_RISK_THRESHOLD

    return pred_df, metrics_df, high_risk_df, rel_con_df


def build_nsdv_rc_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build mean RC table and top-factor summary for NSDV."""
    raw = pd.read_csv(OUTPUT_BASE / "rc" / "NSDV.csv")
    raw = raw.rename(columns={"Key": "risk_factor", "round": "iteration"})
    mean_rc = (
        raw.groupby("risk_factor", as_index=False)["rc"]
        .mean()
        .rename(columns={"rc": "mean_rc"})
        .sort_values("mean_rc", ascending=False)
    )
    mean_rc["Risk factor"] = mean_rc["risk_factor"].map(lambda f: RISK_FACTOR_LABELS.get(f, f))

    ordered = [f for f in RISK_FACTOR_LABELS if f in set(mean_rc["risk_factor"])] + [
        f for f in mean_rc["risk_factor"] if f not in RISK_FACTOR_LABELS
    ]
    order_map = {f: i for i, f in enumerate(ordered)}
    mean_rc["order"] = mean_rc["risk_factor"].map(order_map)
    table = mean_rc.sort_values("order")[["Risk factor", "mean_rc"]].copy()
    table["NSDV"] = table["mean_rc"].map(lambda v: f"{v:.4f}")
    table = table[["Risk factor", "NSDV"]]

    top_factor = mean_rc.sort_values("mean_rc", ascending=False).iloc[0]["risk_factor"]
    top_values = raw.loc[raw["risk_factor"].eq(top_factor), "rc"]
    top_summary = pd.DataFrame([{
        "label": "NSDV", "risk_factor": top_factor,
        "risk_factor_label": RISK_FACTOR_LABELS.get(top_factor, top_factor),
        "mean_rc": top_values.mean(), "sd_rc": top_values.std(ddof=1),
        "variance_rc": top_values.var(ddof=1),
        "mean_percent": top_values.mean() * 100, "sd_percent": top_values.std(ddof=1) * 100,
        "variance_percent_squared": top_values.var(ddof=1) * 10000,
    }])
    return table, top_summary


def export_nsdv_rc_workbook(table: pd.DataFrame, top_summary: pd.DataFrame) -> None:
    RC_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RC_TABLE_DIR / "nsdv_relative_contribution_table.csv",
                 index=False, encoding="utf-8-sig")
    top_summary.to_csv(RC_TABLE_DIR / "nsdv_top_risk_factor_summary.csv",
                       index=False, encoding="utf-8-sig", float_format="%.8f")

    xlsx_path = RC_TABLE_DIR / "nsdv_relative_contribution_tables.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="RC table", index=False)
        top_summary.to_excel(writer, sheet_name="Top factor", index=False)

    wb = load_workbook(xlsx_path)
    ws = wb["RC table"]
    thin = Side(style="thin", color="000000")
    bdr = Border(left=thin, right=thin, top=thin, bottom=thin)
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=2):
        for cell in row:
            cell.border = bdr
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.font = Font(name="Times New Roman", size=12)
    for cell in ws[1]:
        cell.font = Font(name="Times New Roman", size=12, bold=True)
    ws.column_dimensions["A"].width = 44
    ws.column_dimensions["B"].width = 16
    for row_idx in range(1, ws.max_row + 1):
        ws.row_dimensions[row_idx].height = 24


    top_ws = wb["Top factor"]
    for row in top_ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.font = Font(name="Times New Roman", size=11)
    for col_idx in range(1, top_ws.max_column + 1):
        top_ws.column_dimensions[get_column_letter(col_idx)].width = 26
    wb.save(xlsx_path)
    print(f"Saved RC workbook: {xlsx_path}")


# ---------------------------------------------------------------------------
# High-risk area / population summary tables
# ---------------------------------------------------------------------------
POPULATION_PATH = INPUT_DIR / "population.xlsx"
HIGH_RISK_TABLE_DIR = OUTPUT_ROOT / "nsdv_high_risk_tables"


def _load_county_attrs() -> pd.DataFrame:
    names = pd.read_excel(RISK_DATA_PATH, usecols=["gaul2_code", "gaul0_name", "gaul1_name", "gaul2_name"])
    names["gaul2_code"] = names["gaul2_code"].astype(str)
    names = names.drop_duplicates("gaul2_code")
    pop = pd.read_excel(POPULATION_PATH)
    pop["gaul2_code"] = pop["gaul2_code"].astype(str)
    pop = pop[["gaul2_code", "area (km²)", "population density (/km²)"]].copy()
    pop = pop.rename(columns={"area (km²)": "area_km2", "population density (/km²)": "population_density"})
    pop["population"] = (pop["area_km2"] * pop["population_density"]).fillna(0).round().astype("int64")
    attrs = names.merge(pop, on="gaul2_code", how="left")
    for c in ["area_km2", "population_density"]:
        attrs[c] = attrs[c].fillna(0)
    attrs["population"] = attrs["population"].fillna(0).astype("int64")
    return attrs


def _summarize_codes(attrs: pd.DataFrame, codes: pd.Series, range_type: str) -> dict:
    codes = codes.dropna().astype(str).drop_duplicates()
    subset = attrs[attrs["gaul2_code"].isin(codes)]
    area = float(subset["area_km2"].sum())
    pop = int(subset["population"].sum())
    return {
        "range_type": range_type,
        "county_count": int(subset.shape[0]),
        "country_count": int(subset["gaul0_name"].nunique()),
        "area_km2": area, "area_million_km2": area / 1e6,
        "population": pop, "population_million": pop / 1e6,
    }


def export_nsdv_high_risk_tables() -> None:
    attrs = _load_county_attrs()

    # load NSDV prediction
    pred = pd.read_csv(OUTPUT_BASE / "pred" / "NSDV_statistic.csv")
    pred["gaul2_code"] = pred["gaul2_code"].astype(str)
    prob_col = "mean_prob" if "mean_prob" in pred.columns else "mean"
    high_risk = pred[pred[prob_col] >= 0.5]

    # observed NSDV codes
    nsdv_loc = load_nsdv_locations()
    obs_codes = nsdv_loc["gaul2_code"].dropna().astype(str)

    rows = [
        {**_summarize_codes(attrs, obs_codes, "observed"), "label": "NSDV"},
        {**_summarize_codes(attrs, high_risk["gaul2_code"], "predicted_high_risk"), "label": "NSDV"},
    ]
    summary = pd.DataFrame(rows)

    # increase-vs-observed
    metrics = ["county_count", "country_count", "area_km2", "area_million_km2", "population", "population_million"]
    obs = summary[summary["range_type"].eq("observed")].iloc[0]
    prd = summary[summary["range_type"].eq("predicted_high_risk")].iloc[0]
    comp_rows = [{"label": "NSDV"}]
    for m in metrics:
        comp_rows[0][f"observed_{m}"] = obs[m]
        comp_rows[0][f"predicted_{m}"] = prd[m]
        comp_rows[0][f"increase_{m}"] = prd[m] - obs[m]
        comp_rows[0][f"increase_pct_{m}"] = 100 * (prd[m] - obs[m]) / obs[m] if obs[m] else pd.NA
    comparison = pd.DataFrame(comp_rows)

    HIGH_RISK_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(HIGH_RISK_TABLE_DIR / "nsdv_high_risk_summary.csv", index=False, float_format="%.6f")
    comparison.to_csv(HIGH_RISK_TABLE_DIR / "nsdv_high_risk_increase_vs_observed.csv", index=False, float_format="%.6f")
    with pd.ExcelWriter(HIGH_RISK_TABLE_DIR / "nsdv_high_risk_areas.xlsx") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        comparison.to_excel(w, sheet_name="increase_vs_observed", index=False)

    print(f"Saved NSDV high-risk tables: {HIGH_RISK_TABLE_DIR}")
    print(summary.to_string(index=False))


def main() -> None:
    #output files
    for p in [OUTPUT_BASE / "pred", OUTPUT_BASE / "metric", OUTPUT_BASE / "hrc", OUTPUT_BASE / "rc"]:
        p.mkdir(parents=True, exist_ok=True)

    #input files
    #risk factors
    full_data_raw = pd.read_excel(RISK_DATA_PATH)
    full_data = add_model_weights(merge_tick_predictions(full_data_raw))
    feature_cols = get_environment_features(full_data) + TICK_PROB_FEATURES

    #NSDV data
    nsdv_df = load_nsdv_locations()
    nsdv_pos_codes = set(nsdv_df["gaul2_code"].dropna().astype(str).unique())

    #step 1, identify key risk factors
    key_factors, screening_rc = screen_key_factors(full_data, nsdv_pos_codes, feature_cols)
    screening_rc.to_csv(OUTPUT_BASE / "rc" / "NSDV_rc_screening.csv", index=False)

    #step 2, predict use key risk factors
    results = {"pred": [], "metric": [], "hrc": [], "rc": []}
    for simulation in range(ITERATIONS):
        pred_df, metrics_df, high_risk_df, rel_con_df = fit_one_nsdv_model(
            full_data=full_data,
            nsdv_pos_codes=nsdv_pos_codes,
            key_factors=key_factors,
            simulation=simulation,
        )
        results["pred"].append(pred_df)
        results["metric"].append(metrics_df)
        results["hrc"].append(high_risk_df)
        results["rc"].append(rel_con_df)

    #output files
    virus_name = "NSDV"
    for key, frames in results.items():
        if frames:
            final_df = pd.concat(frames, ignore_index=True)
            final_df.to_csv(OUTPUT_BASE / key / f"{virus_name}.csv", index=False)
            if key == "metric":
                _summarize_metrics(final_df, virus_name).to_csv(
                    OUTPUT_BASE / key / f"{virus_name}_statistics.csv",
                    index=False,
                )

    pd.DataFrame({"key_factors": key_factors}).to_csv(OUTPUT_BASE / "key_factors.csv", index=False)

    #calculate mean and 95% CI
    all_preds = pd.concat(results["pred"], ignore_index=True)
    _summarize_predictions(all_preds).to_csv(
        OUTPUT_BASE / "pred" / f"{virus_name}_statistic.csv",
        index=False,
        float_format="%.4f",
    )

    # RC tables
    rc_table, rc_top = build_nsdv_rc_tables()
    export_nsdv_rc_workbook(rc_table, rc_top)
    print(rc_top.to_string(index=False))

    # high-risk area / population summary tables
    export_nsdv_high_risk_tables()



if __name__ == "__main__":
    main()


