# Short-term electricity demand forecasting for New Zealand

Code for my IT8105 Machine Learning project (Assessment 2, Whitecliffe, Trimester 2 2026). The project forecasts New Zealand's national half-hourly grid-supplied electricity demand one day ahead, with a focus on peak demand. It also tests whether the delta-based, peak-weighted target of Bansal (2026), developed for an Indian city, transfers to New Zealand.

Five model families are compared: persistence baselines, Ridge regression, LightGBM, an LSTM and a patch Transformer. Every model is tuned on the same six expanding-window folds of the training period. The winter of 2026 (June to August) is held out as the test set and scored once, after the final model had been chosen from the cross-validation results. A K-means analysis of daily demand profiles adds an unsupervised view of day types.

Every number, table and figure in the report comes from the outputs of these scripts.

## Results

The selected model is the Transformer trained on the change from the same slot yesterday (Δ-slot) with top-decile peak weights, averaged over five seeds. On the winter 2026 test set:

| Model | MAE (MW) | MAPE (%) | Top-decile MAE (MW) |
|---|---|---|---|
| Persistence (same slot, previous day) | 249.1 | 5.22 | 291.1 |
| Ridge | 153.0 | 3.26 | 177.0 |
| LightGBM (Δ-slot, L1) | 118.0 | 2.47 | 133.5 |
| LSTM (Δ-slot, L1) | 115.9 | 2.42 | 122.7 |
| Transformer (absolute target, L1) | 106.8 | 2.26 | 99.3 |
| Transformer (Δ-slot, top-decile weights), selected | 103.0 | 2.19 | 90.2 |

The 95% day-block bootstrap interval for the selected model's MAE is 95.8 to 110.7 MW. The full results, including the Diebold-Mariano tests, are in `out/tables/` after a run.

## Folder structure

```
short-term-electricity-demand-forecasting-nz/
├── README.md                          this file
├── requirements.txt                   Python packages and versions
├── .gitignore                         keeps data, outputs and the virtual environment out of Git
├── selection.json                     copy of the model-selection file from the reported run
├── data_audit.json                    copy of the raw-data audit from the reported run
├── IT8105_A2_full_pipeline.ipynb      the same code as one Jupyter notebook (optional)
├── code/
│   ├── common.py                      shared settings, features, folds, metrics and statistical tests
│   ├── deep.py                        LSTM and Transformer models and their training loop
│   └── 01_build_dataset.py ... 17_report_checks.py   the pipeline steps (see below)
├── data/                              raw input files go here (not in Git)
│   └── processed/                     created by steps 01 and 03
├── out/                               created when the pipeline runs (not in Git)
│   ├── tables/                        results as JSON, CSV and pickle files
│   ├── figs/                          report figures as PNG and PDF
│   └── models/                        saved Ridge, LightGBM, LSTM and Transformer models
└── report/                            created by step 14 (not in Git)
    └── vals_auto.json                 numbers quoted in the report
```

`common.py` and `deep.py` are imported by the step scripts and are not run on their own.

## Data

The data are not included in the repository, as the course requires. Download them and put them directly in `data/`:

1. The 24 monthly `YYYYMM_Grid_export.csv` files from the Electricity Authority's EMI website (emi.ea.govt.nz, wholesale datasets, Grid_export), September 2024 to August 2026. They hold half-hourly metered grid exports in kWh for each point of connection (CC BY 4.0).
2. Hourly weather for three cities from the Open-Meteo Historical Weather API, saved as `Auckland_weather_2024-09_2026-08.csv`, `Wellington_weather_2024-09_2026-08.csv` and `Christchurch_weather_2024-09_2026-08.csv`. Each file has temperature at 2 m, precipitation and wind speed at 10 m.

Step 01 expects the Auckland weather file in fixed UTC+12 and the Wellington and Christchurch files in UTC, which is how they were downloaded. If you download them again with different time-zone settings, change the `offsets` line in `01_build_dataset.py` to match.

The NZ public holiday calendar comes from the `holidays` package, so no file is needed for it.

## Set-up

The reported run used Python 3.13. From the repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Package versions: pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, lightgbm 4.7.0, torch 2.14.1 (CPU), statsmodels 0.15.0, shap 0.54.1, holidays 0.107 and matplotlib 3.11.2, plus scipy and joblib. No GPU is needed.

On macOS, LightGBM (through Homebrew's `libomp`) and PyTorch (which bundles its own `libomp`) can crash with a segmentation fault when both load in one process. This affects steps 08, 10 and 14, and the notebook. Install `libomp` with Homebrew, then point the loader at PyTorch's copy before you run anything:

```bash
brew install libomp
export DYLD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
```

Linux and Windows do not need this.

## Running the pipeline

Run the steps from inside `code/`, in the order below. The numbers in the file names do not match the run order exactly: step 12 (selection) must run before step 08 (test), and step 12 runs step 11 itself.

| Order | Command | What it does | Main outputs |
|---|---|---|---|
| 1 | `python 01_build_dataset.py` | Sums all connection points into national demand, converts kWh to MW, maps trading periods to a 48-slot local-clock grid (daylight-saving fix), aligns the weather time zones and builds population-weighted temperature | `data/processed/national_halfhourly.pkl`, `out/tables/data_audit.json` |
| 2 | `python 02_eda.py` | Exploratory analysis on the cleaned series | `eda.json`, three EDA figures |
| 3 | `python 03_build_tabular.py` | Builds the 25-feature tables (one row per day and slot) and the daily arrays for the neural networks | `data/processed/tab_*.pkl`, `daily_observed_popw.pkl`, `tabular_info.json` |
| 4 | `python 04_tabular_tuning.py` | Cross-validates the persistence baselines, tunes Ridge (alpha grid) and LightGBM (40 random configurations) | `cv_tabular_main.pkl`, `best_lgbm.json`, `tune_*.csv/pkl` |
| 5 | `python 05_deep_tuning.py LSTM 12` | Tunes the LSTM (12 random configurations) | `tune_LSTM.pkl`, `best_LSTM.json` |
| 6 | `python 05_deep_tuning.py Transformer 12` | Tunes the Transformer (12 random configurations) | `tune_Transformer.pkl`, `best_Transformer.json` |
| 7 | `python 06_tabular_ablations.py` | LightGBM ablations: target and loss weighting, λ, weather input, city proxy, feature groups, training length | `ablation_lgbm.pkl` |
| 8 | `python 07_deep_ablations.py LSTM` | LSTM target and loss-weighting ablation | `ablation_LSTM.pkl` |
| 9 | `python 07_deep_ablations.py Transformer` | Transformer target and loss-weighting ablation | `ablation_Transformer.pkl` |
| 10 | `python 12_select.py` | Runs step 11 (CV summaries, corrected t-tests, CV figures), then fixes every selection decision from the CV results | `selection.json`, `ablation_summary.pkl`, `cv_stats.json` |
| 11 | `python 08_final_test.py` | Retrains the selected models on the full training period with five seeds and scores the winter 2026 test set once | `test_results.pkl`, `test_preds.pkl`, `out/models/` |
| 12 | `python 10_test_analysis.py` | Diebold-Mariano tests, bootstrap intervals, error analysis, SHAP and permutation importance, latency and model size | `test_analysis.json`, test figures |
| 13 | `python 13_extra_figures.py` | Learning curve and neural-network training curves | `fig_learning` |
| 14 | `python 09_pipeline_figure.py` | Pipeline diagram (Figure 3.1) | `fig_pipeline` |
| 15 | `python 15_service_figure.py` | Forecasting-service diagram for the application conversion section | `fig_service` |
| 16 | `python 14_vals.py` | Collects the numbers quoted in the report | `report/vals_auto.json`, `vals_tuning.json` |
| 17 | `python 16_weather_and_clusters.py` | Reruns the selected Transformer with persistence weather instead of observed weather, and runs the K-means day-type analysis (after selection, changes no decision) | `weather_and_clusters.json` |
| 18 | `python 17_report_checks.py` | Target standard deviations, tuning cost ratios, peak timing, Holm correction and Newey-West checks of the DM tests | `report_checks.json` |

Files in the outputs column without a folder are in `out/tables/`. Figures are saved to `out/figs/` as both PNG and PDF.

### Run everything with one command

From the repository folder, with the virtual environment active (and `DYLD_LIBRARY_PATH` set on macOS):

```bash
(
  set -e
  cd code
  python 01_build_dataset.py
  python 02_eda.py
  python 03_build_tabular.py
  python 04_tabular_tuning.py
  python 05_deep_tuning.py LSTM 12
  python 05_deep_tuning.py Transformer 12
  python 06_tabular_ablations.py
  python 07_deep_ablations.py LSTM
  python 07_deep_ablations.py Transformer
  python 12_select.py
  python 08_final_test.py
  python 10_test_analysis.py
  python 13_extra_figures.py
  python 09_pipeline_figure.py
  python 15_service_figure.py
  python 14_vals.py
  python 16_weather_and_clusters.py
  python 17_report_checks.py
) 2>&1 | tee run.log
```

The block stops at the first step that fails, and `run.log` keeps a copy of everything printed.

### Running time

The reported run took about 3 hours on an Apple-silicon MacBook Pro using the CPU only, and about 8 hours on a 2-vCPU Linux machine. Most of that time goes to LSTM tuning (step 05) and the neural-network ablations (step 07). Steps 09 to 15 and 17 only read saved outputs, so each finishes in under a minute. Step 16 retrains the selected Transformer ten times (two weather inputs, five seeds), so it takes longer than the other analysis steps.

Each step saves its outputs, so you can rerun one step without repeating the earlier ones. For example, after a full run you can redraw the figures or rerun the test analysis on its own.

### Using the notebook instead

`IT8105_A2_full_pipeline.ipynb` holds the same code as one notebook, with a section for each step in the run order above. The only changes from the scripts are the ones a single notebook needs: the shared modules are defined in the first two cells, steps 05 and 07 loop over the LSTM and Transformer instead of taking command-line arguments, and step 11 runs in its own cell before step 12. Open it from the repository folder (where it sits) so that it finds `data/` and `out/`. On macOS, start Jupyter from a terminal after setting `DYLD_LIBRARY_PATH`, because the notebook loads LightGBM and PyTorch in the same kernel.

The scripts are the reference version: the reported results were produced with them.

## Reproducibility

The random seed is 42 throughout. The neural networks are trained with seeds 42 to 46 and their predictions are averaged.

The test set (1 June to 31 August 2026, 92 days) is used once. Step 12 writes every selection decision to `selection.json` from the cross-validation results before step 08 scores the test set. The `selection.json` in the repository root is the file from the reported run, written at 22:17 NZDT on 30 October 2026, and `data_audit.json` is that run's raw-data audit.

Step 08 saves every seed of each neural network and its scaler statistics, so the five-seed ensemble can be reloaded without retraining. Ridge and LightGBM models are saved with joblib.

Neural-network results can differ slightly between machines even with fixed seeds. A full rerun on the same Mac reproduced every model result exactly; only the timings changed.
