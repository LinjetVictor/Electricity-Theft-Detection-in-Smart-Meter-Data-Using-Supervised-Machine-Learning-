# Electricity Theft Detection in Smart Meter Data Using Supervised Machine Learning

A ranking-based approach to prioritising field inspections.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LinjetVictor/Electricity-Theft-Detection-in-Smart-Meter-Data-Using-Supervised-Machine-Learning-/blob/main/Electricity_Theft_Detection_Colab.ipynb)

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
6. [Feature Engineering](#feature-engineering)
7. [Models](#models)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Results](#results)
10. [Using the Trained Model](#using-the-trained-model)
11. [How to Run](#how-to-run)
12. [Reproducibility](#reproducibility)
13. [Limitations](#limitations)
14. 
---

## Overview

Electricity theft, through meter tampering, illegal connections and billing manipulation, is a major source of non-technical losses for power utilities. It cuts revenue, overloads transformers and feeders, causes outages, and pushes tariffs up for honest customers. Utilities have started rolling out smart meters that record consumption every day, but most still choose customers for inspection by complaints, random sampling or suspicion.

This project builds a **supervised classification model** that learns theft patterns from daily smart meter readings and **ranks customers by theft probability**. A utility with a limited inspection budget can then visit the highest-risk customers first, which raises the share of inspections that find real theft.

**Headline result (held-out test set):** inspecting the 50 highest-risk customers finds theft in **78%** of visits, against a **7.7%** hit rate for random inspection. That is a **10.1x lift**. ROC-AUC is 0.811 and PR-AUC is 0.348.

**Key ideas**

- Convert each customer's raw consumption time series into behavioural features.
- Compare several algorithms under class imbalance (theft is about 7.7% of customers after cleaning).
- Evaluate with metrics that reflect real operations (PR-AUC and **precision@k**) rather than accuracy.
- Explain predictions with SHAP.

---

## Problem Statement

Manual inspection is slow, expensive and inaccurate. Inspection teams are small compared with the number of customers, so most visits are wasted on honest customers while real offenders continue unnoticed. The data to detect theft already exists, but utilities lack an automated and reliable way to turn it into a short, prioritised inspection list. Because theft is rare, simple rules and accuracy-driven models perform poorly: a model that labels everyone as honest would already score above 90% accuracy while catching no thieves.

Theft is commonly expected to leave measurable patterns in consumption data, such as sudden drops in usage, long runs of zero readings, unusually low variance, and consumption far below that of similar customers. These patterns are too numerous for people to check across tens of thousands of meters, but a supervised model can learn which ones matter from customers whose status is already known. 

---

## Objectives

**Main objective:** Build and evaluate a supervised classification model that detects electricity theft from daily smart meter data and ranks customers by risk.

**Specific objectives**

1. Clean the raw smart meter data and engineer meaningful behavioural features from each customer's consumption series.
2. Compare Logistic Regression, Random Forest, XGBoost, LightGBM and a 1D-CNN under class imbalance, using class weighting and SMOTE.
3. Tune the best model with stratified cross-validation and evaluate it once on an untouched test set.
4. Measure performance with precision, recall, F1, ROC-AUC, PR-AUC and precision@k.
5. Explain which behaviours drive the predictions using SHAP.

---

## Dataset

| Property        | Value                                                                            |
|-----------------|----------------------------------------------------------------------------------|
| Name            | SGCC Electricity Theft Detection dataset                                         |
| Original source | State Grid Corporation of China                                                  |
| Where to get it | Kaggle, search for "SGCC electricity theft detection" (uploaded by `bensalem14`) |
| Customers       | 42,372 raw; **31,188 after cleaning**                                            |
| Period          | 1 January 2014 to 31 October 2016 (1,034 daily columns loaded)                   |
| Granularity     | Daily consumption (kWh) per customer                                             |
| Labels          | `FLAG`: 1 = theft, 0 = normal                                                    |
| Theft rate      | 3,615 theft customers (8.5%) raw; 7.7% after cleaning                            |

**Data quality (from the exploration step):** 25.6% of all readings are missing, 13.2% are exact zeros, and there are no negative readings. A customer-level look shows 11,184 customers (26.4%) have more than half of their readings missing, and these were removed during cleaning.

---

## Methodology

The project follows the **CRISP-DM** process: business understanding, data understanding, data preparation, modelling, evaluation and deployment.

| Step                   | Description                                                                                                                                                                                                                              |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. Set up and load     | Install libraries, load the CSV, parse and sort date columns                                                                                                                                                                             |
| 2. Explore (EDA)       | Class balance, missing values, zero and negative readings, consumption curves for theft vs normal customers                                                                                                                              |
| 3. Clean               | Remove duplicates (none found), treat negative readings as missing (none found), drop customers with more than 50% missing data (11,184 dropped), interpolate short gaps (up to 7 days), fill long gaps with the customer's median, cap outliers at each customer's 99th percentile |
| 4. Feature engineering | Convert each time series into one row of 28 behavioural features (see below)                                                                                                                                                             |
| 5. Split               | Stratified 70/15/15 train, validation and test split at customer level, so no customer appears in two sets; add an Isolation Forest anomaly score fitted on training data only (29 features in total)                                     |
| 6. Train and compare   | Train all models with imbalance handling and compare on the validation set                                                                                                                                                               |
| 7. Tune and evaluate   | Randomised search (20 candidates) with stratified 5-fold cross-validation optimising PR-AUC; choose the decision threshold on the validation set; evaluate once on the test set                                                          |
| 8. Explain             | SHAP global and per-customer explanations                                                                                                                                                                                                |

| Split      | Customers | Theft customers | Theft rate |
|------------|----------:|----------------:|-----------:|
| Train      | 21,831    | 1,686           | 7.7%       |
| Validation | 4,678     | 361             | 7.7%       |
| Test       | 4,679     | 362             | 7.7%       |

**Why classification?** Labelled examples of theft and normal behaviour exist, so a supervised model can learn the exact patterns that separate them and output a probability for ranking. Unsupervised anomaly detection alone would flag any unusual customer, such as a vacant house or a seasonal business, so it is used here only as an extra input feature.

**Preventing data leakage**

- The split is done by customer before any model fitting.
- The Isolation Forest, scalers and SMOTE are fitted on training data only. SMOTE runs inside a pipeline so it never touches validation or test data.
- The decision threshold is chosen on the validation set; the test set is used only for the final evaluation.
- Population-level features (e.g., the average consumption curve) use no labels but are computed over all customers, including validation and test.

---

## Feature Engineering

Each customer becomes one row of 28 features, plus the Isolation Forest anomaly score added after the split.

| Group                           | Features                                                                                                                |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Basic statistics                | mean, median, standard deviation, max, min, coefficient of variation                                                    |
| Zero and low usage              | share of zero readings, share of days below half the customer's mean, longest run of zero days, longest run of low days |
| Weekly pattern                  | weekday-to-weekend consumption ratio                                                                                    |
| Seasonality                     | average consumption for each calendar month (12 features)                                                               |
| Trend                           | slope of consumption over time, last quarter vs first quarter ratio                                                     |
| Comparison with other customers | correlation with the population average curve, mean absolute deviation from it, ratio to the population mean            |
| Anomaly score                   | Isolation Forest score (fitted on training data only)                                                                   |

---

## Models

| Model                          | Imbalance handling                            |
|--------------------------------|-----------------------------------------------|
| Logistic Regression (baseline) | Balanced class weights                        |
| Random Forest                  | Balanced subsample class weights              |
| XGBoost                        | `scale_pos_weight` (about 11.9)               |
| LightGBM                       | `scale_pos_weight` (about 11.9)               |
| SMOTE + Random Forest          | SMOTE inside the training pipeline            |
| 1D-CNN                         | Class weights, trained on raw daily sequences scaled to 0-1 per customer, 15 epochs |

XGBoost had the best validation PR-AUC and was tuned with `RandomizedSearchCV` over the number of trees, depth, learning rate, subsampling, column sampling, minimum child weight and gamma.

---

## Evaluation Metrics

Accuracy is not used as the main metric because the data is highly imbalanced.

- **PR-AUC (average precision):** the primary metric for rare-event detection.
- **Precision, recall and F1** at the chosen threshold.
- **ROC-AUC.**
- **Precision@k:** among the k customers ranked highest by predicted risk, the share that are real thieves. This mirrors how inspection teams work, where only k visits can be made. It is reported alongside the random-inspection baseline and the resulting **lift**.
- **Confusion matrix and precision-recall curve** on the test set.

---

## Results

### Model comparison (validation set)

Precision, recall and F1 use a fixed 0.5 threshold; ROC-AUC, PR-AUC and precision@k do not depend on a threshold and are the fairer basis for comparing models.

| Model                 | ROC-AUC | PR-AUC | Precision | Recall | F1    | P@100 | P@500 |
|-----------------------|--------:|-------:|----------:|-------:|------:|------:|------:|
| **XGBoost**           | 0.794   | 0.333  | 0.251     | 0.518  | 0.338 | 0.51  | 0.304 |
| LightGBM              | 0.781   | 0.326  | 0.269     | 0.443  | 0.335 | 0.52  | 0.296 |
| Random Forest         | 0.768   | 0.325  | 0.625     | 0.083  | 0.147 | 0.57  | 0.298 |
| SMOTE + Random Forest | 0.776   | 0.307  | 0.329     | 0.380  | 0.353 | 0.54  | 0.294 |
| Logistic Regression   | 0.733   | 0.243  | 0.161     | 0.607  | 0.255 | 0.40  | 0.238 |
| 1D-CNN                | 0.751   | 0.239  | 0.158     | 0.651  | 0.254 | 0.41  | 0.268 |

The three tree ensembles are close together. XGBoost leads on ROC-AUC and PR-AUC, so it was chosen for tuning. Random Forest looks poor on recall only because 0.5 is a bad threshold for its probability scale; its P@100 is actually the highest. The 1D-CNN on raw sequences did not beat the engineered-feature models, which supports the feature-engineering approach.

### Final tuned model (test set)

Best parameters: `subsample=0.6`, `n_estimators=600`, `min_child_weight=3`, `max_depth=5`, `learning_rate=0.03`, `gamma=0`, `colsample_bytree=1.0`. Best cross-validated PR-AUC on the training data: **0.366**. Decision threshold (maximising F1 on the validation set): **0.673**.

| Metric              | Value |
|---------------------|------:|
| ROC-AUC             | 0.811 |
| PR-AUC              | 0.348 |
| Precision (theft)   | 0.361 |
| Recall (theft)      | 0.406 |
| F1 (theft)          | 0.382 |
| Accuracy            | 0.898 |

Test PR-AUC (0.348) is close to the cross-validated PR-AUC (0.366), so there is no sign of overfitting to the validation set. For context, a random ranking would give a PR-AUC of about 0.077.

### Precision@k (test set)

| Inspections (k) | Precision@k | Random inspection | Lift |
|----------------:|------------:|------------------:|-----:|
| 50              | 0.780       | 0.077             | 10.1 |
| 100             | 0.560       | 0.077             | 7.2  |
| 200             | 0.475       | 0.077             | 6.1  |
| 500             | 0.330       | 0.077             | 4.3  |

In practice: out of the 4,679 test customers, a team that can only make 100 visits would find roughly 56 thieves by following the model, against about 8 with random selection.

### What the model learned

SHAP analysis on the test set shows the strongest drivers of a high theft score are, in order: **maximum daily consumption**, **trend slope**, **longest run of low-usage days**, **correlation with the population average curve** (low correlation raises risk), and **standard deviation**. Weekday-to-weekend ratio and several monthly averages follow.

Some findings differ from the usual textbook picture. In this dataset, theft customers have a **higher** average consumption than normal customers (about 23.5 vs 7.4 kWh/day), a **shorter** longest run of zero readings (about 55 vs 117 days), and a rising rather than falling trend. This is a property of the SGCC data (and of how its labels were produced), so it should not be assumed to carry over to other utilities.

---

## Reproducibility

- A fixed random seed (`RANDOM_STATE = 42`) is used for splits, models and tuning.
- Results can vary slightly between library versions. After a final run, pin versions with `pip freeze > requirements-lock.txt`.
- The full dataset is large (about 156 MB), so check that the pipeline runs on a small sample first.
- Validation results are also saved by the notebook to `model_comparison_validation.csv`.

---

## Limitations

- **Single utility, single region.** The data comes from State Grid Corporation of China, so the model may not transfer to Kenya or other regions without retraining on local data. Real Kenya Power meter data is not publicly available.
- **A quarter of customers were removed.** 11,184 of 42,372 customers (26.4%) had more than half their readings missing and were dropped. If missingness is related to theft, this shifts the theft rate (8.5% to 7.7%) and means the model has not been evaluated on sparse-data customers, who would need a separate approach in production.
- **Modest absolute performance.** At the chosen threshold, about 36% of flagged customers are real thieves and about 41% of thieves are caught. The model is far better than random as a prioritisation tool, but it is not a replacement for inspection.
- **Small test set for top-k estimates.** The test set has 362 theft customers, and precision@50 rests on only 50 customers (39 thieves), so the lift figures at small k carry noticeable uncertainty. No confidence intervals or repeated splits were computed.
- **Dataset-specific patterns.** Theft customers here consume more on average and have shorter zero runs than normal ones, which contradicts common assumptions; other utilities may show different signatures.
- Readings are daily, which hides short-duration tampering that higher-frequency data would show.
- Theft labels come from inspections and may contain errors, and undiscovered thieves are labelled as normal.
- Customer behaviour changes over time, and fraud adapts. The model has not been tested for concept drift, and the split is random by customer rather than by time.
- Some features have extreme values (for example `last_vs_first_quarter` can be very large when early consumption is near zero). Tree models are unaffected, but this can likely affects Logistic Regression.
