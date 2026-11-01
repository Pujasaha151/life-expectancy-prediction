"""
Life Expectancy Prediction
Reconstructed from the user's CSE 4120 Data Mining lab report.

Dataset expected: Life Expectancy Data.csv
Place the CSV in the same folder as this script, then run:
    python main.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Life Expectancy Data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Life expectancy"

# Columns treated for outliers in the report.
OUTLIER_COLUMNS = [
    "Adult Mortality", "infant deaths", "percentage expenditure",
    "Hepatitis B", "Measles", "under-five deaths", "Polio",
    "Total expenditure", "Diphtheria", "HIV/AIDS", "GDP",
    "Population", "thinness 1-19 years", "thinness 5-9 years",
]

# Columns for median imputation in the report.
MEDIAN_COLUMNS = [
    "Adult Mortality", "BMI", "Polio", "thinness 1-19 years",
    "Diphtheria", "thinness 5-9 years",
]


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Download the WHO Life Expectancy dataset and place "
            "'Life Expectancy Data.csv' beside main.py."
        )

    df = pd.read_csv(DATA_FILE)
    # The source dataset sometimes has leading/trailing spaces in headers.
    df.columns = df.columns.str.strip()
    return df


def explore_dataset(df):
    print("\n========== STEP 1: DATASET UNDERSTANDING ==========")
    print("\nFirst five rows:\n", df.head())
    print("\nLast five rows:\n", df.tail())
    print("\nDataset shape:", df.shape)
    print("\nDataset information:")
    df.info()

    print("\n========== STEP 2: DATA QUALITY ==========")
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nMissing-value percentage:\n",
          (df.isnull().sum() / len(df) * 100).round(2))
    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nCategorical value counts:")
    for col in df.select_dtypes(include="object").columns:
        print(f"\n{col}\n{df[col].value_counts(dropna=False)}")

    print("\n========== STEP 3: DISTRIBUTION ==========")
    print("\nNumerical summary:\n", df.describe().T)
    print("\nCategorical summary:\n", df.describe(include="object"))


def create_eda_plots(df):
    print("\nCreating EDA plots in the outputs folder...")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Histograms
    ncols = 3
    nrows = int(np.ceil(len(numeric_cols) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(18, 5 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f"Histogram of {col}")
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "histograms.png", dpi=150)
    plt.close(fig)

    # Boxplots
    nrows = int(np.ceil(len(numeric_cols) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(18, 5 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f"Boxplot of {col}")
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "boxplots_before_treatment.png", dpi=150)
    plt.close(fig)

    # Correlation heatmap
    corr = df.select_dtypes(include=np.number).corr()
    plt.figure(figsize=(14, 11))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.3)
    plt.title("Correlation Matrix of Numerical Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def cap_outliers_using_training_data(X_train, X_test, columns):
    """
    IQR capping based only on training data, avoiding test-set leakage.
    The same lower/upper limits are applied to train and test sets.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    bounds = {}

    for col in columns:
        if col not in X_train.columns:
            continue
        q1 = X_train[col].quantile(0.25)
        q3 = X_train[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        bounds[col] = (lower, upper)
        X_train[col] = X_train[col].clip(lower=lower, upper=upper)
        X_test[col] = X_test[col].clip(lower=lower, upper=upper)

    return X_train, X_test, bounds


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", MinMaxScaler()),
    ])

    # Compatibility with both newer and older scikit-learn releases.
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", drop="first",
                                sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", drop="first",
                                sparse=False)

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline,
                             categorical_features))

    return ColumnTransformer(transformers=transformers,
                             remainder="drop"), numeric_features, categorical_features


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"\n{name} Results")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # Actual vs predicted plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    low = min(float(y_test.min()), float(np.min(predictions)))
    high = max(float(y_test.max()), float(np.max(predictions)))
    plt.plot([low, high], [low, high], "r--")
    plt.xlabel("Actual Life Expectancy (years)")
    plt.ylabel("Predicted Life Expectancy (years)")
    plt.title(f"{name}: Actual vs Predicted")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(OUTPUT_DIR / f"{safe_name}_actual_vs_predicted.png",
                dpi=150)
    plt.close()

    return {"Model": name, "MSE": mse, "RMSE": rmse, "R2": r2}


def main():
    print("Life Expectancy Prediction project starting...")
    df = load_dataset()

    # Remove exact duplicate records if any.
    df = df.drop_duplicates().copy()

    if TARGET not in df.columns:
        raise KeyError(
            f"Target column '{TARGET}' was not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    explore_dataset(df)
    create_eda_plots(df)

    # Rows without a target cannot be used for supervised learning.
    df = df.dropna(subset=[TARGET]).copy()

    # The report's median-imputation step is retained for the named columns.
    # Remaining numeric missing values are handled by KNNImputer in the
    # preprocessing pipeline, fitted only on training data.
    for col in MEDIAN_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(float)

    # Split before learning preprocessing parameters to avoid data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_test, outlier_bounds = cap_outliers_using_training_data(
        X_train, X_test, OUTLIER_COLUMNS
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(
        X_train
    )

    # Save the prepared training/test features for transparency.
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    print("\n========== PREPROCESSING SUMMARY ==========")
    print("Numeric features:", len(numeric_features))
    print("Categorical features:", categorical_features)
    print("Training rows:", X_train_ready.shape[0])
    print("Testing rows:", X_test_ready.shape[0])
    print("Processed feature count:", X_train_ready.shape[1])
    print("Missing values in original data after target removal are handled "
          "by the preprocessing pipeline.")

    # Model 1: Linear Regression
    linear_model = LinearRegression()
    linear_result = evaluate_model(
        "Linear Regression", linear_model,
        X_train_ready, X_test_ready, y_train, y_test
    )

    # Model 2: Decision Tree Regression.
    # The report describes max_depth=5, so it is applied here.
    tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    tree_result = evaluate_model(
        "Decision Tree", tree_model,
        X_train_ready, X_test_ready, y_train, y_test
    )

    results = pd.DataFrame([linear_result, tree_result])
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\n========== MODEL COMPARISON ==========")
    print(results.to_string(index=False))
    print(f"\nFinished. Plots and results are saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
