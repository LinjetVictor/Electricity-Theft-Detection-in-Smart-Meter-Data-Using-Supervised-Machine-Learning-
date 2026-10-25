# Electricity Theft Detection in Smart Meter Data Using Supervised Machine Learning

A ranking-based approach to prioritising field inspections.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/Electricity_Theft_Detection_Colab.ipynb)


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
10. [How to Run](#how-to-run)
11. [Reproducibility](#reproducibility)
12. [Limitations](#limitations)
13. [References](#references)

---

## Overview

Electricity theft, through meter tampering, illegal connections and billing manipulation, is a major source of non-technical losses for power utilities. It cuts revenue, overloads transformers and feeders, causes outages, and pushes tariffs up for honest customers. Utilities have started rolling out smart meters that record consumption every day, but most still choose customers for inspection by complaints, random sampling or suspicion.

This project builds a **supervised classification model** that learns theft patterns from daily smart meter readings and **ranks customers by theft probability**. A utility with a limited inspection budget can then visit the highest-risk customers first, which raises the share of inspections that find real theft.

**Key ideas**

- Convert each customer's raw consumption time series into behavioural features.
- Compare several algorithms under severe class imbalance (theft is about 8.5% of customers).
- Evaluate with metrics that reflect real operations (PR-AUC and **precision@k**) rather than accuracy.
- Explain individual predictions with SHAP.

---

## Problem Statement

Manual inspection is slow, expensive and inaccurate. Inspection teams are small compared with the number of customers, so most visits are wasted on honest customers while real offenders continue unnoticed. The data to detect theft already exists, but utilities lack an automated and reliable way to turn it into a short, prioritised inspection list. Because theft is rare, simple rules and accuracy-driven models perform poorly: a model that labels everyone as honest would already score above 90% accuracy while catching no thieves.

Theft leaves measurable patterns in consumption data, including sudden and sustained drops in usage, long runs of zero readings, unusually low variance, loss of the normal weekday and weekend rhythm, and consumption far below that of similar customers. These patterns are too numerous for people to check across tens of thousands of meters, but a supervised model can learn them from customers whose status is already known.

---

## Objectives

**Main objective:** Build and evaluate a supervised classification model that detects electricity theft from daily smart meter data and ranks customers by risk.

**Specific objectives**

1. Clean the raw smart meter data and engineer meaningful behavioural features from each customer's consumption series.
2. Compare Logistic Regression, Random Forest, XGBoost and LightGBM (and optionally a 1D-CNN) under class imbalance, using class weighting and SMOTE.
3. Tune the best model with stratified cross-validation and evaluate it once on an untouched test set.
4. Measure performance with precision, recall, F1, ROC-AUC, PR-AUC and precision@k.
5. Explain which behaviours drive the predictions using SHAP.

---

## Dataset

| Property | Value |
|---|---|
| Name | SGCC Electricity Theft Detection dataset |
| Original source | State Grid Corporation of China |
| Where to get it | Kaggle, search for "SGCC electricity theft detection" (uploaded by `bensalem14`) |
| Customers | 42,372 |
| Period | January 2014 to October 2016 (1,035 days) |
| Granularity | Daily consumption (kWh) per customer |
| Labels | `FLAG`: 1 = theft, 0 = normal (about 3,615 theft customers, roughly 8.5%) |

---

## Methodology

The project follows the **CRISP-DM** process: business understanding, data understanding, data preparation, modelling, evaluation and deployment.

| Step | Description |
|---|---|
| 1. Set up and load | Install libraries, load the CSV, parse and sort date columns |
| 2. Explore (EDA) | Class balance, missing values, zero and negative readings, consumption curves for theft vs normal customers |
| 3. Clean | Remove duplicates, treat negative readings as missing, drop customers with more than 50% missing data, interpolate short gaps (up to 7 days), fill long gaps with the customer's median, cap outliers at each customer's 99th percentile |
| 4. Feature engineering | Convert each time series into one row of behavioural features (see below) |
| 5. Split | Stratified 70/15/15 train, validation and test split at customer level, so no customer appears in two sets; add an Isolation Forest anomaly score fitted on training data only |
| 6. Train and compare | Train all models with imbalance handling and compare on the validation set |
| 7. Tune and evaluate | Randomised search with stratified 5-fold cross-validation optimising PR-AUC; choose the decision threshold on the validation set; evaluate once on the test set |
| 8. Explain | SHAP global and per-customer explanations |

**Why classification?** Labelled examples of theft and normal behaviour exist, so a supervised model can learn the exact patterns that separate them and output a probability for ranking. Unsupervised anomaly detection alone would flag any unusual customer, such as a vacant house or a seasonal business, so it is used here only as an extra input feature.

**Preventing data leakage**

- The split is done by customer before any model fitting.
- The Isolation Forest, scalers and SMOTE are fitted on training data only. SMOTE runs inside a pipeline so it never touches validation or test data.
- The decision threshold is chosen on the validation set; the test set is used only for the final evaluation.
- Population-level features (e.g., the average consumption curve) use no labels.

---

## Feature Engineering

Each customer becomes one row of features:

| Group | Features |
|---|---|
| Basic statistics | mean, median, standard deviation, max, min, coefficient of variation |
| Zero and low usage | share of zero readings, share of days below half the customer's mean, longest run of zero days, longest run of low days |
| Weekly pattern | weekday-to-weekend consumption ratio |
| Seasonality | average consumption for each calendar month (12 features) |
| Trend | slope of consumption over time, last quarter vs first quarter ratio |
| Comparison with other customers | correlation with the population average curve, mean absolute deviation from it, ratio to the population mean |
| Anomaly score | Isolation Forest score (fitted on training data only) |

---

## Models

| Model | Imbalance handling |
|---|---|
| Logistic Regression (baseline) | Balanced class weights |
| Random Forest | Balanced subsample class weights |
| XGBoost | `scale_pos_weight` |
| LightGBM | `scale_pos_weight` |
| SMOTE + Random Forest | SMOTE inside the training pipeline |
| 1D-CNN (optional) | Class weights, trained on raw daily sequences |

The best-performing model (XGBoost in the notebook) is tuned with `RandomizedSearchCV` over the number of trees, depth, learning rate, subsampling, column sampling, minimum child weight and gamma.

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

> Fill in this section after running the notebook. Use the **test set** numbers from Step 7, and the validation comparison table from Step 6.

**Model comparison (validation set)**

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | P@100 | P@500 |
|---|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | | |
| Random Forest | | | | | | | |
| XGBoost | | | | | | | |
| LightGBM | | | | | | | |
| SMOTE + Random Forest | | | | | | | |

**Final tuned model (test set)**

| Metric | Value |
|---|---|
| ROC-AUC | |
| PR-AUC | |
| Precision | |
| Recall | |
| F1 | |

**Precision@k (test set)**

| Inspections (k) | Precision@k | Random inspection | Lift |
|---|---|---|---|
| 50 | | | |
| 100 | | | |
| 200 | | | |
| 500 | | | |

---

## How to Run

### Option 1: Google Colab 

1. Click the **Open in Colab** badge at the top, or upload `Electricity_Theft_Detection_Colab.ipynb` at [colab.research.google.com](https://colab.research.google.com).
2. Get the data in one of two ways:
   - **Kaggle download:** create a Kaggle API token (Kaggle > Settings > API > Create New Token), upload `kaggle.json` when asked, and run the download cell.
   - **Manual:** download the CSV from Kaggle, upload it to Colab, and place it in a folder named `data/`.
3. Run **Runtime > Run all**. Tuning on the full dataset can take several minutes. Reduce `n_iter` in the search cell if needed.
4. To keep results visible on GitHub, save the notebook **after** running it: **File > Save a copy in GitHub**, or download the `.ipynb` and upload it.

### Option 2: Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Download the dataset from Kaggle and place the CSV in ./data/
jupyter notebook Electricity_Theft_Detection_Colab.ipynb
```

When running locally, remove the cells that import `google.colab` and the lines starting with `!` (Colab-specific commands), or run the equivalent shell commands yourself. The Kaggle CLI can download the data:

```bash
kaggle datasets download -d bensalem14/sgcc-dataset --unzip -p data
```

---

## Reproducibility

- A fixed random seed (`RANDOM_STATE = 42`) is used for splits, models and tuning.
- Results can vary slightly between library versions. Pin versions in `requirements.txt` after your final run (`pip freeze`).
- The full dataset is large, so use the one-month sample on Kaggle first to check that the pipeline runs.

---

## Limitations

- The data comes from a single utility in China, so the model may not transfer to Kenya or other regions without retraining on local data. Real Kenya Power meter data is not publicly available.
- Readings are daily, which hides short-duration tampering that higher-frequency data would show.
- Theft labels come from inspections and may contain errors, and undiscovered thieves are labelled as normal.
- Customer behaviour changes over time, and fraud adapts. The model has not been tested for concept drift.
- Features such as population averages are computed on the whole dataset (without labels), which is mild but worth stating.

---

## References

- Bensalem14. (n.d.). *SGCC electricity theft detection* [Data set]. Kaggle. Original data: State Grid Corporation of China.
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321-357.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE, 10*(3), e0118432.
- Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. *Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining*, 29-39.
- Zheng, Z., Yang, Y., Niu, X., Dai, H.-N., & Zhou, Y. (2018). Wide and deep convolutional neural networks for electricity-theft detection to secure smart grids. *IEEE Transactions on Industrial Informatics, 14*(4), 1606-1615.


