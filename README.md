# Short-term electricity demand forecasting for New Zealand

Code for my IT8105 Machine Learning project (Whitecliffe, Assessment 2). It forecasts New Zealand's national half-hourly grid-supplied demand one day ahead and tests whether the delta-based, peak-weighted target of Bansal (2026) transfers from an Indian city to New Zealand. Persistence, Ridge regression, LightGBM, an LSTM and a patch Transformer are tuned on six expanding-window folds, and the winter of 2026 is held out and scored once.

## Data

The data are not included, as the course requires. Download them and put them in `data/`:

- the 24 monthly `YYYYMM_Grid_export.csv` files from the Electricity Authority's EMI wholesale datasets (September 2024 to August 2026, CC BY 4.0)
- `Auckland_weather_2024-09_2026-08.csv`, `Wellington_weather_2024-09_2026-08.csv` and `Christchurch_weather_2024-09_2026-08.csv` from the Open-Meteo Historical Weather API

## Running it

Run the scripts in this order from `code/`:

    python 01_build_dataset.py      # national DST-corrected series + data audit
    python 02_eda.py
    python 03_build_tabular.py
    python 04_tabular_tuning.py     # persistence, Ridge, LightGBM tuning (6 rolling-origin folds)
    python 05_deep_tuning.py LSTM 12
    python 05_deep_tuning.py Transformer 12
    python 06_tabular_ablations.py
    python 07_deep_ablations.py LSTM
    python 07_deep_ablations.py Transformer
    python 12_select.py             # fixes all selection decisions BEFORE testing (runs 11_cv_figures.py)
    python 08_final_test.py         # scores the held-out winter 2026 once
    python 10_test_analysis.py      # DM tests, bootstrap CIs, error analysis, SHAP, latency
    python 13_extra_figures.py; python 09_pipeline_figure.py; python 15_service_figure.py; python 14_vals.py

Results, figures and saved models go to `out/`. The full run takes about 8 hours on a 2-vCPU machine, mostly LSTM tuning and ablations.

`selection.json` is the selection file, written at 21:32 NZDT on 27 October 2026, before the test set was scored.

## Environment

Python 3.13, pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, lightgbm 4.7.0, torch 2.14.1 (CPU), statsmodels 0.15.0, shap 0.54.1, holidays 0.107 and matplotlib 3.11.2 (see `requirements.txt`). The random seed is 42, and the neural networks are repeated with seeds 42 to 46.
