# Accidents K-Means

A K-Means clustering exercise that groups ~100,000 US traffic accident records by severity and location/distance features using scikit-learn.

## Data

[accidents.csv](accidents.csv) contains 100,000 accident records with time, location (`StartLat`, `StartLng`), weather, and road-feature columns. This exercise clusters on 4 numeric features: `Severity`, `StartLat`, `StartLng`, and `Distance` (length of road affected, in miles).

## What it does

[notebook.ipynb](notebook.ipynb):

1. Selects the 4 numeric features and standardizes them with `StandardScaler`, since K-Means is distance-based and the features are measured on different scales (e.g. lat/lng degrees vs. miles).
2. Runs the **Elbow Method** across `k = 2..10`, plotting inertia (within-cluster sum of squares) to find the point of diminishing returns.
3. Computes the **Silhouette Score** across the same range of `k` to evaluate how well-separated the clusters are, complementing the elbow analysis.
4. Fits a final `KMeans` model with `k=2` and assigns each accident a `Cluster` label.
5. Projects the standardized features to 2D with `PCA` and visualizes the resulting clusters with Plotly.
6. Summarizes each cluster's average feature values, and compares `Severity` and accident counts across clusters.

## Takeaway

The elbow and silhouette analyses support `k=2` as a reasonable number of clusters, splitting accidents primarily along geography and distance rather than severity — severity distributions look similar across clusters, while cluster sizes and locations differ noticeably.

## Figures

The [figures](figures) folder contains the plots generated in the notebook:

- [Elbow Method.png](figures/Elbow%20Method.png)
- [Silhouette Scores.png](figures/Silhouette%20Scores.png)
- [Elbow Method & Silhouette Scores.png](figures/Elbow%20Method%20%26%20Silhouette%20Scores.png)
- [K-Means Clusters (PCA Projection).png](figures/K-Means%20Clusters%20%28PCA%20Projection%29.png)
- [Severity by Cluster.png](figures/Severity%20by%20Cluster.png)
- [Number of Accidents by Cluster.png](figures/Number%20of%20Accidents%20by%20Cluster.png)
