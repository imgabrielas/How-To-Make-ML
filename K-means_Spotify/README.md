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
