import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

project_path = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

dataset_path = os.path.join(
    project_path,
    "Dataset"
)

input_path = os.path.join(
    dataset_path,
    "AirQualityUCI_cleaned.csv"
)

df = pd.read_csv(input_path)

print("=" * 70)
print("1. DATASET LOADED")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)
features = [
    "CO(GT)",
    "C6H6(GT)",
    "NOx(GT)",
    "NO2(GT)",
    "T",
    "RH",
    "AH"
]

X_df = df[features].copy()

print("\n" + "=" * 70)
print("2. CLUSTERING FEATURES")
print("=" * 70)

print("\nSelected features:")
print(features)
print("\nMissing values in selected features:")

print(
    X_df.isnull().sum()
)

X_df = (
    X_df
    .interpolate(
        method="linear",
        limit_direction="both"
    )
    .dropna()
)

print("\nShape after handling missing values:")
print(X_df.shape)

scaler = StandardScaler()

X = scaler.fit_transform(X_df)

print("\n" + "=" * 70)
print("3. FEATURE STANDARDIZATION")
print("=" * 70)

print("\nFeatures standardized successfully.")

# --- FIX 1: Use project_path for Figures ---
figures_path = os.path.join(
    project_path,
    "Figures"
)

os.makedirs(
    figures_path,
    exist_ok=True
)

ks = range(2, 9)
inertias = []
silhouettes = []
for k in ks:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )
    labels = model.fit_predict(X)
    inertias.append(
        model.inertia_
    )
    silhouettes.append(
        silhouette_score(
            X,
            labels
        )
    )

print("\n" + "=" * 70)
print("4. TESTING DIFFERENT NUMBERS OF CLUSTERS")
print("=" * 70)

print("\nK values:")
print(
    list(ks)
)

print("\nInertia values:")
print(
    inertias
)

print("\nSilhouette scores:")
print(
    silhouettes
)

best_k = list(ks)[
    int(
        np.argmax(silhouettes)
    )
]

print("\nSelected K:")
print(best_k)
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

labels = kmeans.fit_predict(X)

X_df["Cluster"] = labels


print("\n" + "=" * 70)
print("5. K-MEANS CLUSTERING RESULTS")
print("=" * 70)
print("\nCluster sizes:")

print(
    X_df["Cluster"]
    .value_counts()
    .sort_index()
)

cluster_profiles = (
    X_df
    .groupby("Cluster")[features]
    .mean()
)

print("\nCluster profiles:")

print(
    cluster_profiles
)

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    list(ks),
    inertias,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia"
)

plt.title(
    "Elbow Method for Selecting K"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        figures_path,
        "elbow_method.png"
    )
)

plt.show()

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    list(ks),
    silhouettes,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "Silhouette Score by Number of Clusters"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        figures_path,
        "silhouette_scores.png"
    )
)

plt.show()

pca = PCA(
    n_components=2
)

X2 = pca.fit_transform(X)

explained_variance = (
    pca.explained_variance_ratio_
)

print("\n" + "=" * 70)
print("6. PCA ANALYSIS")
print("=" * 70)

print("\nExplained variance by PC1:")
print(
    explained_variance[0]
)

print("\nExplained variance by PC2:")
print(
    explained_variance[1]
)

print("\nTotal explained variance:")
print(
    explained_variance.sum()
)

plt.figure(
    figsize=(10, 7)
)

for cluster in sorted(
    X_df["Cluster"].unique()
):
    mask = (
        X_df["Cluster"] == cluster
    )
    plt.scatter(
        X2[mask, 0],
        X2[mask, 1],
        s=10,
        label=f"Cluster {cluster}"
    )

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "K-Means Clusters Visualized with PCA"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        figures_path,
        "kmeans_pca_clusters.png"
    )
)

plt.show()

hierarchical = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward"
)

hierarchical_labels = (
    hierarchical.fit_predict(X)
)

print("\n" + "=" * 70)
print("7. HIERARCHICAL CLUSTERING")
print("=" * 70)

print("\nHierarchical cluster sizes:")

print(
    pd.Series(
        hierarchical_labels
    )
    .value_counts()
    .sort_index()
)

hierarchical_silhouette = (
    silhouette_score(
        X,
        hierarchical_labels
    )
)

print("\nHierarchical clustering silhouette score:")

print(
    hierarchical_silhouette
)

print("\nCreating dendrogram...")

Z = linkage(
    X,
    method="ward"
)

plt.figure(
    figsize=(14, 7)
)

dendrogram(
    Z,
    no_labels=True
)

plt.xlabel(
    "Observations"
)

plt.ylabel(
    "Ward Distance"
)

plt.title(
    "Hierarchical Clustering Dendrogram"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        figures_path,
        "hierarchical_dendrogram.png"
    )
)

plt.show()

plt.figure(
    figsize=(10, 7)
)

for cluster in sorted(
    np.unique(
        hierarchical_labels
    )
):
    mask = (
        hierarchical_labels == cluster
    )
    plt.scatter(
        X2[mask, 0],
        X2[mask, 1],
        s=10,
        label=f"Cluster {cluster}"
    )

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "Hierarchical Clusters Visualized with PCA"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        figures_path,
        "hierarchical_pca_clusters.png"
    )
)

plt.show()

print("\n" + "=" * 70)
print("8. CLUSTERING METHOD COMPARISON")
print("=" * 70)

print("\nK-Means silhouette score:")

kmeans_silhouette = silhouette_score(
    X,
    labels
)

print(
    kmeans_silhouette
)

print("\nHierarchical clustering silhouette score:")

print(
    hierarchical_silhouette
)

clustered_df = df.loc[
    X_df.index
].copy()

clustered_df["KMeans_Cluster"] = labels

clustered_df["Hierarchical_Cluster"] = (
    hierarchical_labels
)


output_path = os.path.join(
    dataset_path,
    "AirQualityUCI_clustered.csv"
)

clustered_df.to_csv(
    output_path,
    index=False
)
output_path = os.path.join(
    dataset_path,
    "AirQualityUCI_clustered.csv"
)

X_df.to_csv(
    output_path,
    index=False
)

# Save cluster profiles
profile_path = os.path.join(
    dataset_path,
    "Cluster_Profiles.csv"
)

cluster_profiles.to_csv(
    profile_path
)


# Save clustering evaluation results
evaluation_path = os.path.join(
    dataset_path,
    "Clustering_Evaluation.csv"
)

evaluation_df = pd.DataFrame({
    "K": list(ks),
    "Inertia": inertias,
    "Silhouette_Score": silhouettes
})

evaluation_df.to_csv(
    evaluation_path,
    index=False
)


# Save PCA cluster results
pca_path = os.path.join(
    dataset_path,
    "PCA_Cluster_Results.csv"
)

pca_results = pd.DataFrame({
    "PC1": X2[:, 0],
    "PC2": X2[:, 1],
    "KMeans_Cluster": labels,
    "Hierarchical_Cluster": hierarchical_labels
})

pca_results.to_csv(
    pca_path,
    index=False
)