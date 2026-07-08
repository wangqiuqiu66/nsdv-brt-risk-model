"""Calculate inverse-probability weights using the VIF workflow.

1. mark counties with any tick or NSDV record as surveyed/status=1;
2. fit a survey-status logistic model using all environmental risk factors;
3. remove variables with VIF >= 10;
4. refit logistic regression and export 1 / predicted probability weights,
   rescaled to mean 1 among surveyed counties.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_ROOT = Path(os.environ.get("BRT_OUTPUT_DIR", PROJECT_ROOT / "output"))
OUTPUT_DIR = OUTPUT_ROOT / "weight_fixed"

RISK_DATA_PATH = INPUT_DIR / "risk_data_all.xlsx"
TICK_LOC_PATH = INPUT_DIR / "tick_location.xlsx"
NSDV_LOC_PATH = INPUT_DIR / "nsdv_location.xlsx"
EXCLUDED_FEATURES = {"Horses_density"}

NON_FEATURE_COLUMNS = {
    "gaul2_code",
    "gaul0_name",
    "gaul1_name",
    "gaul2_name",
    "continent",
    "status",
    "prob",
    "recip",
    "rescaled",
}


def get_recorded_codes(tick_loc: pd.DataFrame, nsdv_loc: pd.DataFrame) -> set[str]:
    tick_codes = tick_loc["gaul2_code"].dropna().astype(str)
    nsdv_codes = nsdv_loc["gaul2_code"].dropna().astype(str)
    return set(tick_codes) | set(nsdv_codes)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    features = [
        col
        for col in df.columns
        if col not in NON_FEATURE_COLUMNS
        and col not in EXCLUDED_FEATURES
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not features:
        raise ValueError("No numeric risk-factor columns were found for the weight model.")
    return features


def load_nsdv_locations() -> pd.DataFrame:
    nsdv_loc = pd.read_excel(NSDV_LOC_PATH, sheet_name="Sheet1")
    required = {"gaul2_code", "is"}
    missing = required.difference(nsdv_loc.columns)
    if missing:
        raise ValueError(f"{NSDV_LOC_PATH} is missing columns: {sorted(missing)}")
    return nsdv_loc.loc[nsdv_loc["is"].ne(0) | nsdv_loc["is"].isna()].copy()


def calculate_vif(scaled_features: pd.DataFrame) -> pd.DataFrame:
    vif_df = pd.DataFrame({"Variable": scaled_features.columns})
    vif_df["VIF"] = [
        variance_inflation_factor(scaled_features.values, i)
        for i in range(scaled_features.shape[1])
    ]
    return vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    risk_data = pd.read_excel(RISK_DATA_PATH)
    tick_loc = pd.read_excel(TICK_LOC_PATH)
    nsdv_loc = load_nsdv_locations()

    recorded_codes = get_recorded_codes(tick_loc, nsdv_loc)
    data = risk_data.copy()
    data["gaul2_code"] = data["gaul2_code"].astype(str)
    data["status"] = data["gaul2_code"].isin(recorded_codes).astype(int)

    feature_cols = get_feature_columns(data)
    raw_features = data[feature_cols]
    if raw_features.isna().any().any():
        missing = raw_features.isna().sum()
        raise ValueError(f"Missing values found in weight-model features: {missing[missing > 0].to_dict()}")

    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(raw_features),
        columns=feature_cols,
        index=data.index,
    )

    vif_df = calculate_vif(scaled_features)
    selected_features = vif_df.loc[vif_df["VIF"] < 10, "Variable"].tolist()
    removed_features = vif_df.loc[vif_df["VIF"] >= 10, "Variable"].tolist()
    if not selected_features:
        raise ValueError("No variables remained after VIF < 10 filtering.")

    x_const = sm.add_constant(scaled_features[selected_features], has_constant="add")
    result = sm.Logit(data["status"], x_const).fit(disp=False)

    coef_pval_df = pd.DataFrame(
        {
            "Variable": result.params.index,
            "Coefficient": result.params.values.round(6),
            "p_value": result.pvalues.values.round(6),
        }
    )

    probabilities = np.asarray(result.predict(x_const), dtype=float)
    data["prob"] = np.clip(probabilities, 1e-8, 1.0)
    survey = data.loc[data["status"].eq(1), ["gaul2_code", "prob"]].copy()
    survey["recip"] = 1.0 / survey["prob"]
    survey["rescaled"] = survey["recip"] / survey["recip"].mean()

    survey.to_excel(OUTPUT_DIR / "weight.xlsx", index=False)
    coef_pval_df.to_excel(OUTPUT_DIR / "logistic_coef_pval.xlsx", index=False)
    vif_df.to_excel(OUTPUT_DIR / "vif_result.xlsx", index=False)
    pd.DataFrame({"selected_features": selected_features}).to_excel(
        OUTPUT_DIR / "selected_vif_features.xlsx",
        index=False,
    )
    pd.DataFrame({"removed_features": removed_features}).to_excel(
        OUTPUT_DIR / "removed_vif_features.xlsx",
        index=False,
    )

    print(f"Feature count before VIF: {len(feature_cols)}")
    print(f"Selected VIF < 10 features: {len(selected_features)}")
    print(f"Recorded counties: {survey.shape[0]}")
    print(f"Saved weights to: {OUTPUT_DIR / 'weight.xlsx'}")


if __name__ == "__main__":
    main()
