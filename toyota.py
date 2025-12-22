# Understand the data (Load Dataset & Basic Inspection)
import pandas as pd
import numpy as np
df = pd.read_csv("toyota.csv")
print("heads",df.head()) 
print("Information",df.info())
print(df.columns)

# Data Quality Analysis (Data cleaning - Missing values & duplicates)
print(df.duplicated().sum())
print(df.isnull().sum())
print(df.isnull().mean() * 100) # Mean percentage of misisng  values

# Descriptive Analysis (shows what is happened)
print("Summary statistics",df.describe()) # Summary statistics
print("rows & coloumns", df.shape)
# Mean price by fuel type
print(df.groupby("fuelType")["price"].mean())
# Average price by car model
print(df.groupby("model")["price"].mean().sort_values(ascending=False))

#EDA (Exploratory Data Analysis- Visualization using matplotlib & seaborn)
import matplotlib
matplotlib.use("TkAgg")   # or "Qt5Agg"
import matplotlib.pyplot as plt

# Price distribution
plt.hist(df["price"], bins=20)
plt.xlabel("Price")
plt.ylabel("Count")
plt.title("Price Distribution")
plt.savefig("plot_name.png", dpi=150, bbox_inches="tight")
plt.show()
print("Second plot is executing")
# Price vs Age
import matplotlib.pyplot as plt
plt.scatter(df["year"], df["price"])
plt.xlabel("year")
plt.ylabel("price")
plt.title("Price vs year")
plt.show()

# Identify numeric columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
print("Numeric columns:", numeric_cols.tolist())
# Correlation heatmap (simple matplotlib version)
corr = df[numeric_cols].corr(numeric_only=True)
plt.figure(figsize=(8, 6))
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.show()
# Histograms for numeric columns
df[numeric_cols].hist(bins=30, figsize=(12, 8))
plt.suptitle("Numeric Feature Distributions", y=1.02)
plt.tight_layout()
plt.show()
# Simple outlier check using IQR for price (prints bounds)
q1, q3 = df["price"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("\n--- Price Outlier Bounds (IQR) ---")
print("Lower bound:", lower, "Upper bound:", upper)

outliers = df[(df["price"] < lower) | (df["price"] > upper)]
print("Outlier rows (by price):", len(outliers))

#  5) Predictive Analytics (Regression: Predict Price)
# Goal: Build a model that predicts price from features.
# Key: Handle categorical variables with OneHotEncoder, numeric with imputation
# ========================
# Scikit-learn Imports
# ========================

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

target = "price"
X = df.drop(columns=[target])
y = df[target]

# Identify columns
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def evaluate(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print(f"\n=== {model_name} ===")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R^2 :", round(r2, 4))

    return preds

# Model 1: Linear Regression baseline
linreg = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LinearRegression())
])
# Model 2: Ridge (often better than plain LinearRegression with one-hot)
ridge = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", Ridge(alpha=1.0, random_state=42))
])
# Model 3: Random Forest (nonlinear, usually strong for tabular data)
rf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])
pred_lin = evaluate("Linear Regression", linreg, X_train, X_test, y_train, y_test)
pred_ridge = evaluate("Ridge Regression", ridge, X_train, X_test, y_train, y_test)
pred_rf = evaluate("Random Forest", rf, X_train, X_test, y_train, y_test)

# Plot predicted vs actual for the best model (often RF)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred_rf, alpha=0.3)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest: Actual vs Predicted")
plt.tight_layout()
plt.show()

# 6.) Feature Importance (Random Forest)
# This is useful to explain "what factors affect price most".
# Need to extract the one-hot feature names after preprocessing.

try:
    # Fit RF if not already fit (safe)
    rf.fit(X_train, y_train)

    # Get feature names
    ohe = rf.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    ohe_feature_names = ohe.get_feature_names_out(cat_features)

    all_feature_names = np.concatenate([num_features, ohe_feature_names])

    importances = rf.named_steps["model"].feature_importances_
    fi = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

    print("\n--- Top 20 Feature Importances (RF) ---")
    print(fi.head(20))

except Exception as e:
    print("\nCould not compute feature importances:", e)



