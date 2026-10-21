# HealthLens: Global Health Metrics Analysis and Mortality Prediction

A data science and machine learning project that analyses global health indicators and predicts key health outcomes. It cleans the WHO Life Expectancy dataset, explores trends across countries, trains tree-based models to predict **Adult Mortality** and **Infant Deaths**, and includes a neural network for predicting **Life Expectancy**.

---

## Objectives

- Analyse global health data to find trends and patterns
- Predict **Adult Mortality** and **Infant Deaths** with machine learning (Random Forest and XGBoost)
- Build a **deep learning model** (Keras neural network) to predict **Life Expectancy**
- Compare traditional ML models with neural networks
- Draw insights on how socioeconomic and healthcare factors relate to population health

---

## Dataset

**Life Expectancy (WHO)** dataset from Kaggle, stored as `Life Expectancy Data.csv`.

| Property | Value |
|----------|-------|
| Rows × columns | 2,938 × 22 |
| Countries | 193 |
| Years | 2000 – 2015 |
| Status split | 2,426 Developing, 512 Developed |
| Missing values (raw) | 2,563 cells across 14 columns (worst: Population 652, Hepatitis B 553, GDP 448) |

**Feature groups:** country, year and development status; life expectancy, adult mortality, infant and under-five deaths; GDP, population and health expenditure; immunisation and disease indicators (Hepatitis B, Polio, Diphtheria, Measles, HIV/AIDS); lifestyle and social factors (alcohol, BMI, thinness, schooling, income composition of resources).

---

## Repository Contents

| File | Purpose |
|------|---------|
| `Life Expectancy Data.csv` | Raw dataset |
| `Data_cleaning.ipynb` | Cleans the raw data and writes `cleaned_health_data.csv` |
| `cleaned_health_data.csv` | Cleaned dataset used for modelling |
| `EDA.ipynb` | Exploratory analysis and visualisations; writes `eda_ready_data.csv` |
| `eda_ready_data.csv` | Identical copy of the cleaned data (byte-for-byte) |
| `Model_training.ipynb` | Trains Random Forest and XGBoost models for Adult Mortality and Infant Deaths |
| `adult_mortality_model.pkl` | Saved XGBoost model for Adult Mortality |
| `infant_deaths_model.pkl` | Saved XGBoost model for Infant Deaths |
| `Life_Expectancy.ipynb` | Neural network for Life Expectancy (see [Known Issues](#known-issues)) |
| `life_expectancy_nn_model.h5` | Saved Keras model from the notebook above |
| `READMD.txt` | Original project notes (superseded by this README) |

---

## Workflow

### 1. Data cleaning (`Data_cleaning.ipynb`)
- Strip and standardise column names (for example `"Life expectancy "` becomes `Life_expectancy`)
- Fill missing numeric values with each column's **median**
- Encode `Status` as `Developing = 0`, `Developed = 1`
- Check for duplicates (none found)
- Save `cleaned_health_data.csv` (0 missing values)

### 2. Exploratory analysis (`EDA.ipynb`)
- Summary statistics and a full correlation heatmap
- Distributions of Adult Mortality and Infant Deaths
- Top 10 countries by average life expectancy
- Global adult mortality trend over time
- GDP vs Life Expectancy, coloured by development status
- Bubble chart of GDP, Life Expectancy, Population and Adult Mortality

### 3. Mortality prediction (`Model_training.ipynb`)
- Drop `Country`; use the remaining columns except the two targets as features (19 features)
- 80/20 train–test split, `random_state=42`
- **Random Forest:** 200 trees
- **XGBoost:** 200 estimators, learning rate 0.05, max depth 6
- XGBoost models are saved with `joblib`

### 4. Life expectancy neural network (`Life_Expectancy.ipynb`)
- Features standardised with `StandardScaler`
- Architecture: Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dense(1)
- Adam optimiser, MSE loss, MAE metric

---

## Results

Test-set performance (20% hold-out):

| Target | Model | MAE | RMSE | R² |
|--------|-------|-----|------|-----|
| Adult Mortality | Random Forest | 49.04 | 78.12 | 0.576 |
| Adult Mortality | XGBoost | 49.52 | 79.47 | 0.561 |
| Infant Deaths | Random Forest | 1.52 | 6.52 | 0.998 |
| Infant Deaths | XGBoost | 1.67 | 8.79 | 0.996 |

Random Forest scores slightly better than XGBoost on both targets, but the XGBoost models are the ones saved to disk.

**Neural network:** the notebook output (MAE 69.25, R² −55.32) comes from a network that was never trained, so it is not a valid result. See below.

---

## Known Issues

1. **The neural network is never trained.** `Life_Expectancy.ipynb` builds and compiles the model but has no `model.fit(...)` call, so the reported metrics and the saved `life_expectancy_nn_model.h5` belong to an untrained network. Adding a training step (for example `model.fit(X_train, y_train, epochs=100, validation_split=0.2)`) before evaluation, then re-saving, is needed before any comparison with the tree models.
2. **Infant Deaths R² is inflated by a near-duplicate feature.** `under_five_deaths` stays in the feature set and correlates 0.997 with `infant_deaths`, so the 0.996–0.998 scores mostly reflect that relationship rather than genuine prediction from socioeconomic factors. Dropping it would give a more meaningful result.
3. **Possible data leakage in preprocessing.** Median imputation runs on the whole dataset before the train–test split, and the split is random by row, so the same country appears in both training and test sets. A country-wise split would give a more realistic estimate of generalisation.
4. **Documentation mismatch.** `READMD.txt` (filename typo included) describes a `data/raw`, `data/processed`, `notebooks/` layout with numbered notebooks. The actual folder is flat and uses different notebook names.

---

## Setup

Python 3 (the notebooks were run on Python 3.12) with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib tensorflow
```

Run the notebooks in this order from inside the `DL_Project` folder (they read files by relative path):

1. `Data_cleaning.ipynb`
2. `EDA.ipynb`
3. `Model_training.ipynb`
4. `Life_Expectancy.ipynb`

### Using the saved mortality models

Features must match the training columns and order:

```python
import joblib
import pandas as pd

df = pd.read_csv("cleaned_health_data.csv")
X = df.drop(columns=["Country", "Adult_Mortality", "infant_deaths"])

adult_model = joblib.load("adult_mortality_model.pkl")
print(adult_model.predict(X.head()))
```

Load `.pkl` files only from sources you trust.

---

## Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · XGBoost · TensorFlow/Keras · joblib · Jupyter
