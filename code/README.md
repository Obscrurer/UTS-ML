# Code — UTS Machine Learning Project

This folder contains the three independent Google Colab notebooks:
- `01_delivery_time_regression.ipynb`
- `02_fraud_classification.ipynb`
- `03_customer_clustering.ipynb`

## Run instructions

1. Upload `data_uts_ml.zip` to Google Drive at `MyDrive/UTS_ML/data_uts_ml.zip`.
2. Open the relevant notebook in Google Colab.
3. Run all cells from top to bottom.
4. The notebooks load the relevant data files from the extracted archive and generate their prediction CSV.

The clustering notebook writes `cluster.csv` to match the required submission filename. The regression and classification notebooks write `prediksi_regresi.csv` and `prediksi_klasifikasi.csv`.

## Reproduction caveat

The three notebooks are independent pipelines and require the original dataset archive plus their notebook runtime dependencies. Because the course data archive is not included and the notebooks use Google Colab/Drive setup, this repository does not currently provide a single local shell command that executes all three pipelines end-to-end. To reproduce the report's numbers, run each notebook from top to bottom in Colab. The committed report and CSVs are the outputs from the recorded runs.

## Environment

The notebooks install or use CatBoost and HDBSCAN where required. Common libraries include Python, NumPy, pandas, scikit-learn, matplotlib, and seaborn.
