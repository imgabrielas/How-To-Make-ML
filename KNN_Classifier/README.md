# KNN Classifier

A simple K-Nearest Neighbors classification exercise using scikit-learn.

## Data

[data.csv](data.csv) contains 20 samples with 3 numeric features (`feature1`, `feature2`, `feature3`) and a binary `label` (0 or 1). The two classes form two well-separated clusters, making them easy to visualize and classify.

## What it does

[exercise.ipynb](exercise.ipynb):

1. Loads and parses the semicolon-separated data into a DataFrame.
2. Visualizes the 3 features in an interactive 3D scatter plot with Plotly (`plotly.express.scatter_3d`), colored by label.
3. Trains a `KNeighborsClassifier` on the data and predicts labels for new, unseen points, plotting the predictions alongside the training data.
4. Repeats the prediction with a different `n_neighbors` value to show how the choice of `k` affects classification results.

## Takeaway

Changing `n_neighbors` from 1 to 4 changed the predicted labels for the same new points, illustrating how sensitive KNN is to the choice of `k`.