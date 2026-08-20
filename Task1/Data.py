import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
project_path = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
# Path to original dataset
input_path = os.path.join(
    project_path,
    "Dataset",
    "AirQualityUCI.csv"
)
# Load dataset
df = pd.read_csv(
    input_path,
    sep=";",
    decimal=","
)

print("DATASET LOADED SUCCESSFULLY")


df = df.dropna(axis=1, how="all")
df = df.dropna(axis=0, how="all")

print("\nShape after removing empty rows and columns:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nOriginal Missing Values:")
print(df.isnull().sum())


print("\nData Types:")
print(df.dtypes)

numeric_cols = df.select_dtypes(
    include=np.number
).columns

missing_code = (
    df[numeric_cols] == -200
).sum()

print("\nNumber of -200 values in each column:")

print(
    missing_code[missing_code > 0]
)

df[numeric_cols] = df[numeric_cols].replace(
    -200,
    np.nan
)

print("\nMissing values after replacing -200 with NaN:")

print(
    df.isnull().sum()
)

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

df["Time"] = pd.to_datetime(
    df["Time"],
    format="%H.%M.%S",
    errors="coerce"
).dt.time

print("\nProcessed Date and Time:")
print(
    df[["Date", "Time"]].head()
)

df["DateTime"] = pd.to_datetime(
    df["Date"].astype(str)
    + " "
    + df["Time"].astype(str),
    errors="coerce"
)

print("\nDateTime column:")
print(
    df[["Date", "Time", "DateTime"]].head()
)

missing = df.isnull().sum()

print("\nColumns containing missing values:")

print(
    missing[missing > 0]
)

plt.figure(
    figsize=(12, 6)
)

sns.heatmap(
    df.isnull(),
    cbar=False
)

plt.title(
    "Missing Values in the Dataset"
)

plt.xlabel(
    "Columns"
)

plt.ylabel(
    "Rows"
)

plt.tight_layout()

plt.show()
numeric_cols = df.select_dtypes(
    include=np.number
).columns

df[numeric_cols] = df[numeric_cols].interpolate(
    method="linear"
)

df[numeric_cols] = (
    df[numeric_cols]
    .ffill()
    .bfill()
)

print("\nMissing values after treatment:")

print(
    df.isnull().sum()
)

Q1 = df[numeric_cols].quantile(
    0.25
)

Q3 = df[numeric_cols].quantile(
    0.75
)

IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = (
    (df[numeric_cols] < lower)
    |
    (df[numeric_cols] > upper)
)

print("\nNumber of outliers in each numerical column:")

print(
    outliers.sum()
)

plt.figure(
    figsize=(10, 6)
)

sns.boxplot(
    data=df[
        [
            "CO(GT)",
            "NOx(GT)",
            "NO2(GT)"
        ]
    ]
)

plt.title(
    "Outlier Detection Using Boxplots"
)

plt.xlabel(
    "Pollutants"
)

plt.ylabel(
    "Values"
)

plt.tight_layout()

plt.show()

duplicate_count = df.duplicated().sum()

print(
    "\nDuplicate rows before cleaning:",
    duplicate_count
)

df = df.drop_duplicates()

print(
    "Duplicate rows after cleaning:",
    df.duplicated().sum()
)

print("\nFinal dataset shape:")

print(
    df.shape
)

print("\nFinal dataset information:")

df.info()

print("\nFinal missing values:")

print(
    df.isnull().sum()
)

print("\nFinal statistical summary:")

print(
    df.describe()
)

output_path = os.path.join(
    project_path,
    "Dataset",
    "AirQualityUCI_cleaned.csv"
)

df.to_csv(
    output_path,
    index=False
)


print("CLEANED DATASET SAVED SUCCESSFULLY")


print("\nSaved file:")
print(output_path)

print("\nFinal dataset shape:")
print(df.shape)