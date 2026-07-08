This repository contains the cleaned scripts for running Boosted Regression Tree (BRT) analyses of tick-vector and Nairobi sheep disease virus (NSDV) risk.

The BRT workflow was converted from notebook-style analyses into standalone Python scripts. All required libraries can be installed using the packages listed in "requirements.txt". The scripts read input files from the local "input/" directory and write generated results to the local "output/" directory by default.

Contents:

(1) "brt_scripts/01_weight.py": Calculates inverse-probability weights for surveyed GAUL2 areas. NSDV observations are read from "input/nsdv_location.xlsx" (Sheet1), and rows where "is == 0" are excluded.

(2) "brt_scripts/02_tick.py": Runs BRT models for the three tick vectors, exports model metrics, predicted probabilities, high-risk counties, and relative-contribution tables.

(3) "brt_scripts/03_nsdv.py": Runs the NSDV BRT model using environmental variables and predicted probabilities from the three tick-vector models.

(4) "brt_scripts/04_tick_pdp.py" and "brt_scripts/05_nsdv_pdp.py": Calculate and plot partial dependence curves for key tick-vector and NSDV risk factors.

(5) "brt_scripts/06_pop_area.py": Summarizes observed and predicted high-risk counties, area, and population size for tick vectors and NSDV.

(6) "input/" directory: Includes the required input files "risk_data_all.xlsx", "tick_location.xlsx", "nsdv_location.xlsx", and "population.xlsx".

(7) "output/" directory: Stores generated weights, model outputs, prediction summaries, relative-contribution tables, partial-dependence data, and population/area summaries.

Recommended running order:

```powershell
conda activate nsdv_code
python brt_scripts\01_weight.py
python brt_scripts\02_tick.py
python brt_scripts\03_nsdv.py
python brt_scripts\04_tick_pdp.py
python brt_scripts\05_nsdv_pdp.py
python brt_scripts\06_pop_area.py
```

Optional settings:

(1) Set "BRT_OUTPUT_DIR" to write results to a different output directory.

(2) Set "BRT_WEIGHT_MODE" to "none", "no_weight", "unweighted", or "onweight" to run models without inverse-probability weights.

Tick and NSDV high-risk GAUL2 areas are defined using "mean_prob >= 0.5" in the summary outputs. Tick and NSDV relative-contribution tables are exported as decimal values with four digits after the decimal point.

If you encounter any bugs or issues with the code, please feel free to get in touch!
