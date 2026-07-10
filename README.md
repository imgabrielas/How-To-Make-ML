# How To Make ML

A collection of small, self-contained machine learning exercises. Each
subdirectory is an specific exercise focused on one model —
implemented from scratch to build understanding of how it works.

No shared code or dependencies between projects.

## Projects

| Project | Model | Description |
|---|---|---|
| [neuron/](neuron/) | Logistic regression (single neuron) | A single artificial neuron implemented from scratch with NumPy, trained with gradient descent, with an animated visualization of the decision boundary converging during training. |
| [KNN_Classifier/](KNN_Classifier/) | K-Nearest Neighbors | A simple 3-feature, binary-label classification exercise using scikit-learn's `KNeighborsClassifier`, with an interactive 3D Plotly visualization and a comparison of predictions across different `k` values. |
| [K-means_Spotify/](K-means_Spotify/) | K-Means Clustering | Clusters ~32,000 Spotify tracks by 12 standardized audio features using scikit-learn's `KMeans`, choosing `k` via the elbow method and silhouette score, then visualizing clusters with a PCA projection. |
| [K-means_Accidents/](K-means_Accidents/) | K-Means Clustering | Clusters ~100,000 US traffic accident records by severity and location/distance features using scikit-learn's `KMeans`, choosing `k` via the elbow method and silhouette score, then visualizing clusters with a PCA projection. |

More model-focused projects will be added here over time, each in its own subdirectory.