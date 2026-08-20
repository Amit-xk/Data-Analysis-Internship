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

input_path = os.path.join(
    project_path,
    "Dataset",
    "AirQualityUCI_cleaned.csv"
)

df = pd.read_csv(input_path)

print("1. DATASET LOADED")


print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("2. INITIAL DATA ANALYSIS")


print("\nDataset information:")
df.info()

print("\nDescriptive statistics:")
print(df.describe())

print("3. DATE AND TIME PROCESSING")


# Convert Date
df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

# Convert Time
df["Time"] = pd.to_datetime(
    df["Time"],
    format="%H:%M:%S",
    errors="coerce"
).dt.time

# Create DateTime
df["DateTime"] = pd.to_datetime(
    df["Date"].astype(str)
    + " "
    + df["Time"].astype(str),
    errors="coerce"
)

# Sort by DateTime
df = df.sort_values("DateTime")

print("\nDate and Time:")
print(
    df[
        [
            "Date",
            "Time",
            "DateTime"
        ]
    ].head()
)

print("4. CREATING TIME FEATURES")


df["Year"] = df["DateTime"].dt.year
df["Month"] = df["DateTime"].dt.month
df["Month_Name"] = df["DateTime"].dt.month_name()
df["Day"] = df["DateTime"].dt.day
df["Hour"] = df["DateTime"].dt.hour
df["Day_Name"] = df["DateTime"].dt.day_name()

print("\nNew time-based columns:")
print(
    df[
        [
            "DateTime",
            "Year",
            "Month",
            "Month_Name",
            "Day",
            "Hour",
            "Day_Name"
        ]
    ].head()
)

print("5. MISSING VALUE ANALYSIS")


missing_values = df.isnull().sum()

print("\nMissing values:")
print(
    missing_values[missing_values > 0]
)

print("\nTotal missing values:")
print(
    df.isnull().sum().sum()
)

pollution_columns = [
    "CO(GT)",
    "NMHC(GT)",
    "C6H6(GT)",
    "NOx(GT)",
    "NO2(GT)"
]

print("6. BASIC STATISTICAL ANALYSIS")


print("\nPollution statistics:")
print(
    df[pollution_columns].describe()
)

figures_path = os.path.join(
    project_path,
    "Figures"
)

os.makedirs(
    figures_path,
    exist_ok=True
)

print("7. CO DISTRIBUTION")


plt.figure(figsize=(10, 6))
sns.histplot(df["CO(GT)"], bins=30, kde=True)
plt.title("Distribution of Carbon Monoxide (CO)")
plt.xlabel("CO Concentration")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "co_distribution.png"))
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df["NOx(GT)"], bins=30, kde=True)
plt.title("Distribution of Nitrogen Oxides (NOx)")
plt.xlabel("NOx Concentration")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "nox_distribution.png"))
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]])
plt.title("Distribution and Variability of Major Pollutants")
plt.xlabel("Pollutants")
plt.ylabel("Concentration")
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "pollutant_boxplot.png"))
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df["DateTime"], df["CO(GT)"], label="CO")
plt.title("Carbon Monoxide Concentration Over Time")
plt.xlabel("Date")
plt.ylabel("CO Concentration")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "co_over_time.png"))
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df["DateTime"], df["NOx(GT)"], label="NOx")
plt.title("NOx Concentration Over Time")
plt.xlabel("Date")
plt.ylabel("NOx Concentration")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "nox_over_time.png"))
plt.show()

monthly_pollution = df.groupby("Month")[pollution_columns].mean()

print("8. MONTHLY POLLUTION ANALYSIS")


print("\nAverage pollution by month:")
print(monthly_pollution)

plt.figure(figsize=(10, 6))
plt.plot(monthly_pollution.index, monthly_pollution["CO(GT)"], marker="o", label="CO")
plt.title("Average CO Concentration by Month")
plt.xlabel("Month")
plt.ylabel("Average CO Concentration")
plt.xticks(range(1, 13))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "monthly_co.png"))
plt.show()

hourly_pollution = df.groupby("Hour")[pollution_columns].mean()

print("9. HOURLY POLLUTION ANALYSIS")


print("\nAverage pollution by hour:")
print(hourly_pollution)

plt.figure(figsize=(12, 6))
plt.plot(hourly_pollution.index, hourly_pollution["CO(GT)"], marker="o", label="CO")
plt.title("Average CO Concentration by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average CO Concentration")
plt.xticks(range(0, 24))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "hourly_co.png"))
plt.show()

print("10. CORRELATION ANALYSIS")


correlation_columns = [
    "CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH"
]

correlation_matrix = df[correlation_columns].corr()

print("\nCorrelation matrix:")
print(correlation_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Air Quality Variables")
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "correlation_matrix.png"))
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="CO(GT)", y="NOx(GT)")
plt.title("Relationship Between CO and NOx")
plt.xlabel("CO Concentration")
plt.ylabel("NOx Concentration")
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "co_vs_nox.png"))
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="T", y="CO(GT)")
plt.title("Relationship Between Temperature and CO")
plt.xlabel("Temperature")
plt.ylabel("CO Concentration")
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "temperature_vs_co.png"))
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="NOx(GT)", y="NO2(GT)")
plt.title("Relationship Between NOx and NO2")
plt.xlabel("NOx Concentration")
plt.ylabel("NO2 Concentration")
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "nox_vs_no2.png"))
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df["T"], bins=30, kde=True)
plt.title("Distribution of Temperature")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "temperature_distribution.png"))
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df["RH"], bins=30, kde=True)
plt.title("Distribution of Relative Humidity")
plt.xlabel("Relative Humidity (%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "humidity_distribution.png"))
plt.show()

average_pollution = df[pollution_columns].mean()

print("16. AVERAGE POLLUTION LEVELS")


print("\nAverage concentration:")
print(average_pollution)

plt.figure(figsize=(10, 6))
average_pollution.sort_values().plot(kind="bar")
plt.title("Average Concentration of Major Pollutants")
plt.xlabel("Pollutant")
plt.ylabel("Average Concentration")
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "average_pollution.png"))
plt.show()

print("17. EXTREME POLLUTION OBSERVATIONS")


print("\nHighest CO observations:")
print(
    df[["DateTime", "CO(GT)", "NOx(GT)", "NO2(GT)"]]
    .sort_values("CO(GT)", ascending=False)
    .head(10)
)

print("18. FINAL EDA SUMMARY")


print("\nDataset shape:")
print(df.shape)
print("\nNumber of observations:")
print(len(df))
print("\nNumber of variables:")
print(len(df.columns))

print("\nDate range:")
print(df["DateTime"].min())
print("to")
print(df["DateTime"].max())

print("\nAverage pollutant values:")
print(df[pollution_columns].mean())

print("\nCorrelation of CO with other variables:")
print(correlation_matrix["CO(GT)"].sort_values(ascending=False))

output_path = os.path.join(
    project_path,
    "Dataset",
    "AirQualityUCI_EDA.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\nEDA dataset saved successfully to:")
print(output_path)

print("\n" + "=" * 70)
print("WEEK 2 EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal columns:")
print(df.columns.tolist())

print("\nEDA dataset location:")
print(output_path)