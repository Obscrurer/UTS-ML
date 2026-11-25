# UTS Machine Learning Project — Final Report

## Executive summary

This project implements three independent machine-learning workflows: delivery-time regression, transaction-fraud classification, and customer clustering. Each task has its own notebook and prediction file.

| Task | Method | Main result |
|---|---|---|
| Delivery-time regression | CatBoost quantile regression | 60,000 test predictions with P50 and P90 estimates |
| Fraud classification | CatBoost classifier + Platt calibration + cost-based review rule | 106,674 test predictions; 865 transactions flagged |
| Customer clustering | HDBSCAN | 2 clusters and 36 noise observations across 120,000 customers |

The results are not equally conclusive. Regression has chronological holdout metrics; fraud classification exhibits substantial temporal drift, especially in August; clustering is highly stable under the recorded subsampling experiment but yields a broad active/inactive split rather than many nuanced personas.

---

## Task 1 — Delivery-time regression

### Objective and data

Predict `durasi_jam` for packages. The supplied data contains 250,000 training rows and 60,000 test rows. The target is modeled at two quantiles:
- **P50:** median delivery-duration estimate.
- **P90:** upper-quantile estimate, intended to represent a slower-delivery scenario.

The training period covers January–August 2025; the test period covers September–October 2025.

### Validation design and model

A chronological split was used to avoid training on future observations:
- Training: 1 January–30 June 2025 (181,396 rows).
- Validation: 1 July–31 August 2025 (68,604 rows).

A constant-quantile baseline was compared with CatBoost quantile models. Internal early stopping used a separate June period for the model-selection workflow. The final selected iteration counts were 519 for P50 and 230 for P90. Production models were then refit on all 250,000 training records.

### Validation results

| Metric | Constant baseline | CatBoost |
|---|---:|---:|
| P50 pinball loss | 17.3965 | 10.7363 |
| P90 pinball loss | 16.0098 | 11.3706 |
| P50 empirical coverage | 0.4868 | 0.4558 |
| P90 empirical coverage | 0.8920 | 0.8692 |

CatBoost improves pinball loss for both quantiles relative to the constant baseline. Empirical P50 and P90 coverage are below their nominal quantiles on the July–August validation set, so the predictions should not be described as perfectly calibrated prediction intervals.

A June-derived additive calibration reduced June pinball loss and brought June coverage close to target, but its transfer to July–August worsened both pinball losses and coverage. The notebook therefore rejects that calibration for production.

### Test output

`prediksi_regresi.csv` contains 60,000 rows and columns `id_paket`, `p50`, and `p90`. The saved output has no missing values or duplicate package IDs, all predictions are finite and positive, and no P50/P90 crossing remains.

Mean test predictions:
- P50: 69.8008 hours.
- P90: 94.7607 hours.

**Limitation:** test labels are not available in the supplied materials, so test-period accuracy cannot be measured here.

---

## Task 2 — Fraud classification

### Objective and data

Estimate `is_fraud` probability and decide which transactions should be flagged for manual review. Training contains 403,077 transactions; test contains 106,674. Fraud prevalence in the training data is approximately 0.6505%.

The cost assumptions used in the notebook are:
- Each manually reviewed transaction costs Rp25,000.
- A flagged fraud transaction is held and does not incur the modeled fraud loss.
- An unflagged fraud transaction incurs loss equal to its nominal transaction amount.

### Data audit and leakage controls

The workflow audits missingness, temporal ordering, same-user/same-timestamp events, and candidate feature timing. `flag_review_manual` was excluded because it is strongly associated with the target and may reflect a post-review decision. A `missing_history` indicator was added for the two history variables with substantial missingness.

The model uses 82 candidate features: six categorical and 76 numerical, including the engineered missing-history indicator. The notebook also compares variants without `v15` and without user/merchant IDs; those alternatives did not consistently improve the temporal validation results.

### Temporal validation and model

Chronological partitions:
- January–May: 247,464 rows; 1,737 fraud cases.
- June: 50,375 rows for early stopping.
- July: 52,334 rows for policy selection/calibration.
- August: 52,904 rows for final temporal validation.

A class-weighted CatBoost classifier was selected with best iteration 605. July Platt scaling improved Brier score:
- July: 0.003839 raw to 0.002634 calibrated.
- August: 0.006128 raw to 0.004462 calibrated.

### Ranking performance and temporal drift

| Period | PR-AUC | ROC-AUC | Fraud cases |
|---|---:|---:|---:|
| June | 0.71420 | 0.98638 | 299 |
| July | 0.72229 | 0.98775 | 341 |
| August | 0.29620 | 0.95852 | 245 |

August performance degrades materially. The notebook identifies a strong distribution shift in `v15` (PSI 1.937 in August) and changing missing-history rates. Removing `v15` did not improve August PR-AUC (0.28999 versus 0.29620), and removing user/merchant IDs produced only a small August improvement (0.29982) while worsening June/July. The selected baseline is retained, but the drift remains a major limitation.

### Cost-policy evaluation on August

The dynamic review rule flags a transaction when estimated fraud probability multiplied by nominal amount exceeds Rp25,000.

| Policy | Flagged | Missed fraud | Total modeled cost |
|---|---:|---:|---:|
| No review | 0 | 245 | Rp250,001,200 |
| Review everything | 52,904 | 0 | Rp1,322,600,000 |
| Dynamic rule | 396 | 123 | Rp119,168,200 |

On the August validation period, the dynamic rule reduces modeled cost by approximately 52.3% relative to no review. This is an offline estimate under the stated cost assumptions, not a guarantee of operational savings.

### Test output

`prediksi_klasifikasi.csv` contains 106,674 rows with `id_transaksi`, `prob`, and `tandai`. It has no missing values or duplicate transaction IDs. There are 865 flagged transactions (0.811%); predicted probabilities range from approximately 0.001316 to 0.896489.

**Limitations:** September–October test labels are unavailable, so test performance is unknown. August PR-AUC deterioration means probability thresholds and review capacity should be monitored and recalibrated before deployment.

---

## Task 3 — Customer clustering

### Objective and data

Segment 120,000 customers using customer-level summaries, including RFM behavior, category and time-of-day compositions, promotion/payment behavior, engagement, demographics, and 15 anonymous numeric features (`n01`–`n15`).

### Preprocessing

- The dataset contains 6,736 missing `rata2_keranjang` values. All correspond to customers with zero purchases and zero spending in the 90-day window. These were filled with zero, and `tidak_aktif_90h` was added as an explicit inactivity indicator.
- Strongly right-skewed nonnegative variables were transformed with `log1p`.
- Category shares (16 columns) and time-of-day shares (4 columns) were transformed using centered log-ratio (CLR) with a small pseudocount.
- Numerical features were robust-scaled. Feature values were winsorized at the 0.5th and 99.5th percentiles before scaling.
- `perangkat` was one-hot encoded; customer IDs were excluded from the feature matrix and retained for the output.
- Category and time composition groups were weighted by the square root of their feature counts to moderate their aggregate contribution.

### Algorithm comparison

On a 10,000-customer sample:

| Method | Setting | Silhouette / other evidence |
|---|---|---|
| MiniBatch K-Means | k=4 | Best sampled silhouette: 0.5177 |
| Gaussian Mixture | k=2 | Silhouette 0.4954; BIC decreased through k=7 in the tested range |
| HDBSCAN | `min_cluster_size=50`, `min_samples=10` | 3 clusters in the small sample; silhouette 0.5147 excluding noise |

The small-sample method comparison did not by itself settle the final choice. K-Means showed low subsampling stability (ARI 0.1633 for two 80% samples) and mean pairwise seed ARI 0.6391. HDBSCAN produced ARI 0.9999 when two independent 20% samples were compared on their shared 4,882 customers, with two clusters in both samples and approximately 0.1% noise in each.

The final full-data HDBSCAN fit took approximately 30.51 minutes and returned:
- Cluster 0: 113,232 customers (94.36%).
- Cluster 1: 6,732 customers (5.61%).
- Noise (`-1`): 36 customers (0.03%).

Removing the engineered inactivity indicator and refitting HDBSCAN produced ARI 1.0000 against the original labels, with only one customer changing to noise. This indicates that the active/inactive split is also represented by other behavioral features.

### Cluster profiles and recommended actions

**Cluster 0 — Active customers (113,232; 94.36%)**
- Median recency: 9.60 days.
- Median 90-day frequency: 7.
- Median 90-day spending: Rp1,261,000.
- Mean category shares are highest in food and beverages (14.9%), fashion (14.2%), beauty (9.1%), and household products (9.1%).
- Recommended actions: loyalty and repeat-purchase programs, cross-selling, and category recommendations based on observed purchase mix. The mean shopping-time share is highest at night (28.4%), followed by morning (25.9%).

**Cluster 1 — Inactive customers (6,732; 5.61%)**
- Median recency: 38.45 days.
- Median 90-day frequency and spending: zero.
- All customers in this cluster have `tidak_aktif_90h = 1`.
- Recommended actions: controlled win-back campaigns, reminders, and category-personalized reactivation offers. Prioritize previously observed category interests rather than treating all inactive customers identically.

**Noise (`-1`) — Unassigned (36; 0.03%)**
- Keep as a separate operational label rather than forcing assignment.
- Monitor individually or assign through a documented fallback policy if downstream systems require a non-noise segment.

### Limitations

The selected solution is highly stable in the reported subsampling experiment, but it is dominated by an active/inactive lifecycle distinction. It should be described as a robust two-group lifecycle segmentation, not as a discovery of many detailed personas. HDBSCAN stability results are specific to the chosen feature engineering, metric, and hyperparameters.

---

## Reproducibility and deliverables

Each task is maintained in an independent notebook under `notebooks/`. The prediction files are under `predictions/`.

The notebooks expect the original `data_uts_ml.zip` archive in Google Drive at `MyDrive/UTS_ML/`. The archive itself is not included in this repository. Run each notebook from top to bottom in Google Colab.

Prediction artifacts:
- `prediksi_regresi.csv` — 60,000 rows.
- `prediksi_klasifikasi.csv` — 106,674 rows.
- `prediksi_clustering.csv` — 120,000 rows.

## Overall conclusion

All three tasks have completed their prediction-generation workflows. The regression model improves substantially over a constant quantile baseline on chronological validation. The fraud classifier demonstrates useful ranking and cost-policy performance in July/August validation but suffers meaningful temporal drift that requires monitoring. HDBSCAN produces a highly stable two-cluster lifecycle segmentation with a very small noise group. These caveats should accompany any presentation or deployment of the results.
