# Spotify K-Means

A K-Means clustering exercise that groups ~32,000 Spotify tracks by audio characteristics using scikit-learn.

## Data

[spotify.csv](spotify.csv) contains 32,833 tracks with metadata (track/artist/album/playlist info) and 12 numeric audio features: `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, and `duration_ms`.

## What it does

[notebook.ipynb](notebook.ipynb):

1. Selects the 12 numeric audio features and standardizes them with `StandardScaler`, since K-Means is distance-based and the features are measured on very different scales (e.g. 0–1 for `danceability` vs. hundreds of thousands for `duration_ms`).
2. Runs the **Elbow Method** across `k = 2..10`, plotting inertia (within-cluster sum of squares) to find the point of diminishing returns.
3. Computes the **Silhouette Score** across the same range of `k` to evaluate how well-separated the clusters are, complementing the elbow analysis.
4. Fits a final `KMeans` model with `k=4` and assigns each track a `Cluster` label.
5. Projects the standardized features to 2D with `PCA` and visualizes the resulting clusters with Plotly.
6. Summarizes each cluster's average audio features, and compares `track_popularity` and track counts across clusters.

## Takeaway

The elbow and silhouette analyses both support `k=4` as a reasonable number of clusters. The resulting groups separate tracks along recognizable audio dimensions (e.g. high-energy/danceable vs. acoustic/instrumental), and track popularity varies noticeably across clusters — showing that unsupervised grouping on audio features alone picks up patterns that correlate with how tracks perform.

## Figures

The [figures](figures) folder contains the plots generated in the notebook:

- [Figure1 Elbow Method for Choosing the Number of Clusters.png](figures/Figure1%20Elbow%20Method%20for%20Choosing%20the%20Number%20of%20Clusters.png)
- [Figure 2 Silhouette Scores for Different Numbers of Clusters.png](figures/Figure%202%20Silhouette%20Scores%20for%20Different%20Numbers%20of%20Clusters.png)
- [Figure 3 K-Means Clusters (PCA Projection).png](figures/Figure%203%20K-Means%20Clusters%20%28PCA%20Projection%29.png)
- [Figure 4 Track Popularity by Cluster.png](figures/Figure%204%20Track%20Popularity%20by%20Cluster.png)
- [Figure 5 Number of Songs in Each Cluster.png](figures/Figure%205%20Number%20of%20Songs%20in%20Each%20Cluster.png)