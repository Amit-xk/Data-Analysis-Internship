import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import (train_test_split, KFold, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LinearRegression, Ridge)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)

project_path = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

dataset_path = os.path.join(
    project_path,
    "Dataset"
)

file_path = os.path.join(
    dataset_path,
    "AirQualityUCI_cleaned.csv"
)

df = pd.read_csv(file_path)

print("=" * 70)
print("1. DATASET LOADED")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\n" + "=" * 70)
print("2. FEATURE ENGINEERING")
print("=" * 70)

# Convert Date to datetime
df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

# Convert Time to datetime
df["Time_dt"] = pd.to_datetime(
    df["Time"],
    format="%H:%M:%S",
    errors="coerce"
)

# If the cleaned dataset contains Time in
# HH:MM:SS format, extract the hour.
df["Hour"] = df["Time_dt"].dt.hour

# Extract month
df["Month"] = df["Date"].dt.month

# Extract day of week
df["DayOfWeek"] = df["Date"].dt.dayofweek

print("\nNew features created:")
print(
    [
        "Hour",
        "Month",
        "DayOfWeek"
    ]
)

print("\n" + "=" * 70)
print("3. PREDICTION PROBLEM")
print("=" * 70)

print("\nProblem:")
print(
    "Predict Carbon Monoxide (CO) concentration "
    "using other air-quality and environmental variables."
)

# Target variable
target = "CO(GT)"

# Features used for prediction
features = [
    "C6H6(GT)",
    "NOx(GT)",
    "NO2(GT)",
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
    "Hour",
    "Month",
    "DayOfWeek"
]

print("\nTarget variable:")
print(target)

print("\nPredictor variables:")
print(features)

data = df[
    features + [target]
].copy()
data = data.dropna(
    subset=[target]
)

X = data[features]
y = data[target]

print("\n" + "=" * 70)
print("4. DATA PREPARATION")
print("=" * 70)

print("\nFeature matrix shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nMissing values in features:")
print(X.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 70)
print("5. TRAIN-TEST SPLIT")
print("=" * 70)

print("\nTraining samples:")
print(len(X_train))

print("\nTesting samples:")
print(len(X_test))

models = {
    "Linear Regression": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LinearRegression()
        )
    ]),
    "Ridge Regression": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            Ridge(
                alpha=1.0
            )
        )
    ]),
    "Random Forest": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                max_features="sqrt"
            )
        )
    ])
}

print("\n" + "=" * 70)
print("6. MODEL TRAINING AND EVALUATION")
print("=" * 70)

results = []

for name, model in models.items():
    print("\n" + "-" * 50)
    print(name)
    print("-" * 50)

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Make predictions
    predictions = model.predict(
        X_test
    )

    # Calculate MAE
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    # Calculate RMSE
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    # Calculate R2
    r2 = r2_score(
        y_test,
        predictions
    )

    results.append(
        [
            name,
            mae,
            rmse,
            r2
        ]
    )

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2"
    ]
)

results_df = results_df.sort_values(
    "RMSE"
)

print("\n" + "=" * 70)
print("7. MODEL COMPARISON")
print("=" * 70)

print(results_df)

results_path = os.path.join(
    dataset_path,
    "Model_Comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]

best_predictions = best_model.predict(
    X_test
)

print("\n" + "=" * 70)
print("8. BEST MODEL")
print("=" * 70)

print("\nBest model:")
print(best_name)

print("\n" + "=" * 70)
print("9. 5-FOLD CROSS-VALIDATION")
print("=" * 70)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_mae = -cross_val_score(
    best_model,
    X,
    y,
    cv=kf,
    scoring="neg_mean_absolute_error"
).mean()

cv_rmse = -cross_val_score(
    best_model,
    X,
    y,
    cv=kf,
    scoring="neg_root_mean_squared_error"
).mean()

cv_r2 = cross_val_score(
    best_model,
    X,
    y,
    cv=kf,
    scoring="r2"
).mean()

print("\n5-Fold CV MAE:")
print(cv_mae)

print("\n5-Fold CV RMSE:")
print(cv_rmse)

print("\n5-Fold CV R2:")
print(cv_r2)

figures_path = os.path.join(
    project_path,
    "Figures"
)

os.makedirs(
    figures_path,
    exist_ok=True
)

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.4
)

mn = min(
    y_test.min(),
    best_predictions.min()
)

mx = max(
    y_test.max(),
    best_predictions.max()
)

plt.plot(
    [mn, mx],
    [mn, mx]
)

plt.xlabel(
    "Actual CO"
)

plt.ylabel(
    "Predicted CO"
)

plt.title(
    "Actual vs Predicted CO"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        figures_path,
        "actual_vs_predicted_CO.png"
    )
)

plt.show()

residuals = (
    y_test -
    best_predictions
)

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    best_predictions,
    residuals,
    alpha=0.4
)

plt.axhline(
    0
)

plt.xlabel(
    "Predicted CO"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Residual Analysis"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        figures_path,
        "residual_analysis.png"
    )
)

plt.show()

if best_name == "Random Forest":
    importance = pd.Series(
        best_model
        .named_steps["model"]
        .feature_importances_,
        index=features
    ).sort_values(
        ascending=False
    )

    print("\n" + "=" * 70)
    print("10. RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 70)

    print("\nFeature importance:")
    print(
        importance
    )

    importance_path = os.path.join(
        dataset_path,
        "Feature_Importance.csv"
    )

    importance.to_csv(
        importance_path,
        header=["Importance"]
    )

    # Plot feature importance
    plt.figure(
        figsize=(10, 7)
    )

    importance.sort_values().plot(
        kind="barh"
    )

    plt.xlabel(
        "Importance"
    )

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            figures_path,
            "random_forest_feature_importance.png"
        )
    )

    plt.show()

prediction_results = pd.DataFrame({
    "Actual_CO": y_test.values,
    "Predicted_CO": best_predictions,
    "Residual": residuals.values
})

prediction_path = os.path.join(
    dataset_path,
    "CO_Predictions.csv"
)

prediction_results.to_csv(
    prediction_path,
    index=False
)

print("\n" + "=" * 70)
print("11. FINAL SUPERVISED LEARNING SUMMARY")
print("=" * 70)

print("\nTarget:")
print(target)

print("\nNumber of observations:")
print(len(data))

print("\nNumber of features:")
print(len(features))

print("\nBest model:")
print(best_name)

print("\nTest MAE:")
print(
    results_df.iloc[0]["MAE"]
)

print("\nTest RMSE:")
print(
    results_df.iloc[0]["RMSE"]
)

print("\nTest R2:")
print(
    results_df.iloc[0]["R2"]
)

print("\n5-Fold CV MAE:")
print(cv_mae)

print("\n5-Fold CV RMSE:")
print(cv_rmse)

print("\n5-Fold CV R2:")
print(cv_r2)

print("\nModel comparison saved to:")
print(results_path)

print("\nPrediction results saved to:")
print(prediction_path)

print("\nFigures saved in:")
print(figures_path)

