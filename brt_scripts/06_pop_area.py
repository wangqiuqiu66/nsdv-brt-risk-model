"""Summarize predicted high-risk counties, area, and population size."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = Path(os.environ.get("BRT_OUTPUT_DIR", PROJECT_ROOT / "output"))

#files
POPULATION_PATH = INPUT_DIR / "population.xlsx"
TICK_LOCATION_PATH = INPUT_DIR / "tick_location.xlsx"
NSDV_LOCATION_PATH = INPUT_DIR / "nsdv_location.xlsx"

#tick data
TICKS = [
    "Rhipicephalus appendiculatus",
    "Haemaphysalis intermedia",
    "Haemaphysalis longicornis",
]


def find_column(df: pd.DataFrame, prefix: str) -> str:
    """Find a population-table column by prefix to avoid unit-symbol issues."""

    #match columns by prefix
    matches = [column for column in df.columns if str(column).startswith(prefix)]
    return matches[0]


def load_population() -> pd.DataFrame:
    """Load the GAUL2 area and population-density table."""

    #population data
    df = pd.read_excel(POPULATION_PATH)
    df["gaul2_code"] = df["gaul2_code"].astype(str)
    df["gaul0_name"] = df["gaul0_name"].astype(str)
    df["gaul1_name"] = df["gaul1_name"].astype(str)
    df["gaul2_name"] = df["gaul2_name"].astype(str)

    area_col = find_column(df, "area")
    density_col = find_column(df, "population density")
    #calculate population size
    df["population"] = (df[area_col] * df[density_col]).fillna(0).astype("int64")
    return df


def normalize_codes(codes) -> list[str]:
    #county codes
    return pd.Series(codes).dropna().astype(str).drop_duplicates().tolist()


def summarize_codes(population: pd.DataFrame, codes) -> tuple[int, float, int]:
    """Return county count, area, and population for a set of GAUL2 codes."""

    #county count, area and population
    subset = population[population["gaul2_code"].isin(normalize_codes(codes))]
    area_col = find_column(subset, "area")
    return (
        subset.shape[0],
        subset[area_col].sum(),
        int(subset["population"].sum()),
    )


def choose_probability_column(df: pd.DataFrame) -> str:
    for column in ["mean_prob", "mean", "pred"]:
        if column in df.columns:
            return column
    raise ValueError("Prediction summary must contain one of: mean_prob, mean, pred")


def load_nsdv_locations() -> pd.DataFrame:
    nsdv_location = pd.read_excel(NSDV_LOCATION_PATH, sheet_name="Sheet1")
    required = {"gaul2_code", "is"}
    missing = required.difference(nsdv_location.columns)
    if missing:
        raise ValueError(f"{NSDV_LOCATION_PATH} is missing columns: {sorted(missing)}")
    return nsdv_location.loc[
        nsdv_location["is"].ne(0) | nsdv_location["is"].isna()
    ].copy()


def format_count(predicted: int, observed: int) -> str:
    #predicted / observed county count
    ratio = 100 * (predicted - observed) / observed if observed > 0 else np.nan
    return f"{predicted}/{observed} ({ratio:.1f})"


def format_scaled(predicted: float, observed: float, scale: float = 1e6) -> str:
    #predicted / observed area or population
    pred_scaled = predicted / scale
    obs_scaled = observed / scale
    ratio = 100 * (pred_scaled - obs_scaled) / obs_scaled if observed > 0 else np.nan
    return f"{pred_scaled:.2f}/{obs_scaled:.2f} ({ratio:.1f})"


def calculate_tick_summary() -> pd.DataFrame:
    """Compare observed and predicted high-risk ranges for tick vectors."""

    #input files
    population = load_population()
    tick_location = pd.read_excel(TICK_LOCATION_PATH)
    tick_location["gaul2_code"] = tick_location["gaul2_code"].astype(str)

    rows = []
    for tick in TICKS:
        #observed tick counties
        observed_codes = tick_location.loc[tick_location["Species"].eq(tick), "gaul2_code"]
        observed = summarize_codes(population, observed_codes)

        #predicted high-risk counties by mean predicted probability
        pred_path = (
            OUTPUT_DIR
            / "host"
            / tick
            / "unified_result"
            / "prediction"
            / f"{tick}_prediction_summary.csv"
        )
        pred_summary = pd.read_csv(pred_path)
        pred_summary["gaul2_code"] = pred_summary["gaul2_code"].astype(str)
        prob_col = choose_probability_column(pred_summary)
        predicted_codes = pred_summary.loc[pred_summary[prob_col] >= 0.5, "gaul2_code"]
        predicted = summarize_codes(population, predicted_codes)

        rows.append(
            [
                tick,
                format_count(predicted[0], observed[0]),
                format_scaled(predicted[1], observed[1]),
                format_scaled(predicted[2], observed[2]),
            ]
        )

    #output files
    result = pd.DataFrame(rows, columns=["tick", "county", "area", "population"])
    result.to_excel(OUTPUT_DIR / "PC_tick.xlsx", index=False)
    return result


def calculate_nsdv_summary() -> pd.DataFrame:
    """Compare observed and predicted high-risk ranges for NSDV."""

    #input files
    population = load_population()
    nsdv_location = load_nsdv_locations()
    nsdv_location["gaul2_code"] = nsdv_location["gaul2_code"].astype(str)

    #observed NSDV counties
    observed = summarize_codes(population, nsdv_location["gaul2_code"])

    #predicted high-risk counties
    pred_summary_path = OUTPUT_DIR / "NSDV" / "pred" / "NSDV_statistic.csv"
    if pred_summary_path.exists():
        pred_summary = pd.read_csv(pred_summary_path)
        prob_col = "mean_prob" if "mean_prob" in pred_summary.columns else "mean"
        predicted_codes = pred_summary.loc[pred_summary[prob_col] >= 0.5, "gaul2_code"]
    else:
        hrc = pd.read_csv(OUTPUT_DIR / "NSDV" / "hrc" / "NSDV.csv")
        predicted_codes = hrc["gaul2_code"]
    predicted = summarize_codes(population, predicted_codes)

    #output files
    result = pd.DataFrame(
        [
            [
                "NSDV",
                format_count(predicted[0], observed[0]),
                format_scaled(predicted[1], observed[1]),
                format_scaled(predicted[2], observed[2]),
            ]
        ],
        columns=["virus", "county", "area", "population"],
    )
    result.to_excel(OUTPUT_DIR / "PC_nsdv.xlsx", index=False)
    return result


def calculate_future_nsdv_summary(threshold: float = 0.5) -> pd.DataFrame:
    """Summarize future NSDV predictions when future outputs are available."""

    #current high-risk counties
    population = load_population()
    current = pd.read_csv(OUTPUT_DIR / "NSDV" / "hrc" / "NSDV.csv")
    current = current[current["pred"] >= threshold] if "pred" in current.columns else current
    current_stats = summarize_codes(population, current["gaul2_code"])

    #future high-risk counties
    future_dir = OUTPUT_DIR / "future_NSDV" / "future_hrc"
    rows = []
    for path in sorted(future_dir.glob("*.csv")):
        future = pd.read_csv(path)
        if "pred" in future.columns:
            future = future[future["pred"] >= threshold]
        future_stats = summarize_codes(population, future["gaul2_code"])
        scenario = path.stem.replace("NSDV_", "").replace("_Future_HRC_All", "")
        rows.append(
            [
                scenario,
                format_count(future_stats[0], current_stats[0]),
                format_scaled(future_stats[1], current_stats[1]),
                format_scaled(future_stats[2], current_stats[2]),
            ]
        )

    #output files
    result = pd.DataFrame(rows, columns=["scenario", "county", "area", "population"])
    result.to_excel(OUTPUT_DIR / "PC_future_nsdv.xlsx", index=False)
    return result


if __name__ == "__main__":
    #calculate current summaries
    calculate_tick_summary()
    calculate_nsdv_summary()

