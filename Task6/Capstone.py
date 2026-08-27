import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,classification_report,precision_score,recall_score,f1_score,roc_auc_score,silhouette_score)
import torch
import torch.nn as nn
import torch.optim as optim

print("CAPSTONE PROJECT: TELCO CUSTOMER CHURN PREDICTION")

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(project_path, "Dataset")
figures_path = os.path.join(project_path, "Figures")

os.makedirs(dataset_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

raw_file_path = os.path.join(
    dataset_path, "Telco_Customer_Churn_Raw.csv"
)
clean_file_path = os.path.join(
    dataset_path, "Telco_Customer_Churn_Cleaned.csv"
)

url = (
    "https://raw.githubusercontent.com/treselle-systems/"
    "customer_churn_analysis/master/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

if not os.path.exists(raw_file_path):
    print("Downloading dataset from public repository...")
    df = pd.read_csv(url)
    df.to_csv(raw_file_path, index=False)
    print("Download complete!")
else:
    print("Loading dataset from local folder...")
    df = pd.read_csv(raw_file_path)

print(f"\nInitial Dataset Shape: {df.shape}")
print("\nCleaning data...")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

missing_values = df["TotalCharges"].isna().sum()

if missing_values > 0:
    print(
        f"Found {missing_values} missing values in TotalCharges. "
        "Imputing with median..."
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df.to_csv(clean_file_path, index=False)

print("\nData Cleaning Complete!")
print(f"Cleaned Dataset Shape: {df.shape}")
print(f"Saved cleaned dataset to: {clean_file_path}")

print("\n3. EXPLORATORY DATA ANALYSIS & VISUALIZATION")

df_clean = pd.read_csv(clean_file_path)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(6, 5))
ax = sns.countplot(
    data=df_clean,
    x="Churn",
    hue="Churn",
    palette=["#4C72B0", "#DD8452"],
    legend=False
)
plt.title(
    "Overall Customer Churn Distribution",
    fontsize=14,
    fontweight="bold"
)
plt.xlabel("Churn (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)

for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.0f}",
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha="center",
        va="baseline",
        fontsize=11,
        color="black",
        xytext=(0, 5),
        textcoords="offset points"
    )

plt.tight_layout()
plt.savefig(
    os.path.join(figures_path, "capstone_churn_distribution.png"),
    dpi=300
)
plt.show() 
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(
    data=df_clean,
    x="Contract",
    hue="Churn",
    palette=["#4C72B0", "#DD8452"]
)
plt.title(
    "Customer Churn by Contract Type",
    fontsize=14,
    fontweight="bold"
)
plt.xlabel("Contract Type", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.legend(title="Churn", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig(
    os.path.join(figures_path, "capstone_churn_by_contract.png"),
    dpi=300
)
plt.show()  
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_clean,
    x="tenure",
    y="MonthlyCharges",
    hue="Churn",
    palette=["#4C72B0", "#DD8452"],
    alpha=0.5,
    s=20
)
plt.title(
    "Tenure vs Monthly Charges (Colored by Churn)",
    fontsize=14,
    fontweight="bold"
)
plt.xlabel("Tenure (Months)", fontsize=12)
plt.ylabel("Monthly Charges ($)", fontsize=12)
plt.legend(title="Churn", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig(
    os.path.join(figures_path, "capstone_tenure_vs_charges.png"),
    dpi=300
)
plt.show()  
plt.close()

print(f"EDA visualizations generated and saved to: {figures_path}")

print("\n4. UNSUPERVISED LEARNING (K-MEANS CLUSTERING)")

cluster_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

X_cluster = df_clean[cluster_features].copy()

scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df_clean["Customer_Segment"] = kmeans.fit_predict(
    X_cluster_scaled
)

silhouette = silhouette_score(
    X_cluster_scaled,
    df_clean["Customer_Segment"]
)

print("K-Means clustering complete.")
print("Silhouette Score:", round(silhouette, 4))
print("Customers in each segment:")
print(
    df_clean["Customer_Segment"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df_clean,
    x="tenure",
    y="MonthlyCharges",
    hue="Customer_Segment",
    palette="viridis",
    alpha=0.6,
    s=30
)
plt.title(
    "Customer Segmentation (K-Means Clustering)",
    fontsize=14,
    fontweight="bold"
)
plt.xlabel("Tenure (Months)", fontsize=12)
plt.ylabel("Monthly Charges ($)", fontsize=12)
plt.legend(title="Segment")
plt.tight_layout()
plt.savefig(
    os.path.join(figures_path, "capstone_kmeans_clusters.png"),
    dpi=300
)
plt.show()  
plt.close()

print(
    "Cluster visualization saved to:",
    os.path.join(figures_path, "capstone_kmeans_clusters.png")
)

df_encoded = pd.get_dummies(
    df_clean.drop(
        "Customer_Segment",
        axis=1,
        errors="ignore"
    ),
    drop_first=True,
    dtype=float
)

X = df_encoded.drop("Churn", axis=1).values
y = df_encoded["Churn"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler_ml = StandardScaler()

X_train_scaled = scaler_ml.fit_transform(X_train)
X_test_scaled = scaler_ml.transform(X_test)

print("\n5. SUPERVISED LEARNING (RANDOM FOREST CLASSIFIER)")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train_scaled, y_train)

rf_predictions = rf_model.predict(X_test_scaled)
rf_probabilities = rf_model.predict_proba(X_test_scaled)[:, 1]

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_precision = precision_score(
    y_test,
    rf_predictions
)

rf_recall = recall_score(
    y_test,
    rf_predictions
)

rf_f1 = f1_score(
    y_test,
    rf_predictions
)

rf_auc = roc_auc_score(
    y_test,
    rf_probabilities
)

print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"F1 Score: {rf_f1:.4f}")
print(f"ROC-AUC: {rf_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))

print("\n6. DEEP LEARNING (PYTORCH NEURAL NETWORK)")

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)

X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)


class ChurnNN(nn.Module):

    def __init__(self, input_dim):
        super(ChurnNN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


torch.manual_seed(42)

nn_model = ChurnNN(
    input_dim=X_train_tensor.shape[1]
)

criterion = nn.BCELoss()

optimizer = optim.Adam(
    nn_model.parameters(),
    lr=0.001
)

print("Training Neural Network for 50 epochs...")

epochs = 50

for epoch in range(epochs):

    nn_model.train()

    optimizer.zero_grad()

    outputs = nn_model(X_train_tensor)

    loss = criterion(
        outputs,
        y_train_tensor
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {loss.item():.4f}"
        )

nn_model.eval()

with torch.no_grad():

    nn_preds_prob = nn_model(
        X_test_tensor
    )

    nn_predictions = (
        nn_preds_prob >= 0.5
    ).float().numpy().flatten()

nn_accuracy = accuracy_score(
    y_test,
    nn_predictions
)

nn_precision = precision_score(
    y_test,
    nn_predictions
)

nn_recall = recall_score(
    y_test,
    nn_predictions
)

nn_f1 = f1_score(
    y_test,
    nn_predictions
)

nn_auc = roc_auc_score(
    y_test,
    nn_preds_prob.numpy().flatten()
)

print(f"\nNeural Network Accuracy: {nn_accuracy * 100:.2f}%")
print(f"Precision: {nn_precision:.4f}")
print(f"Recall: {nn_recall:.4f}")
print(f"F1 Score: {nn_f1:.4f}")
print(f"ROC-AUC: {nn_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, nn_predictions))

results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "PyTorch Neural Network"
    ],
    "Accuracy": [
        rf_accuracy,
        nn_accuracy
    ],
    "Precision": [
        rf_precision,
        nn_precision
    ],
    "Recall": [
        rf_recall,
        nn_recall
    ],
    "F1_Score": [
        rf_f1,
        nn_f1
    ],
    "ROC_AUC": [
        rf_auc,
        nn_auc
    ]
})

results_path = os.path.join(
    dataset_path,
    "Model_Evaluation_Results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print("\nModel results saved to:", results_path)
