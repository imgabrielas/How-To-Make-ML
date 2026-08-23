# t-SNE

A t-SNE (t-distributed Stochastic Neighbor Embedding) exercise that projects high-dimensional body-measurement data into 2D to explore whether visible clusters emerge.

t-SNE is a nonlinear dimensionality-reduction technique for visualizing high-dimensional data: it models pairwise similarities between points and finds a low-dimensional (2D/3D) layout where similar points stay close together and dissimilar ones stay far apart. Unlike PCA, it captures nonlinear structure but its axes have no direct interpretation, and results can vary between runs.

## Data

`ansure_male.csv` (gitignored) is the ANSUR body-measurement survey, ~1,986 records with 99 columns of anthropometric measurements (heights, circumferences, lengths, etc.) plus categorical fields (`Branch`, `Component`, `Gender`) and derived fields (`BMI`, `BMI_class`, `Height_class`).

## What it does

[exercise.ipynb](exercise.ipynb):

1. Loads the dataset and inspects which columns are non-numeric (`object` dtype).
2. Drops the non-numeric columns (`Branch`, `Component`, `Gender`, `BMI_class`, `Height_class`) into `df_num`, keeping only numeric body-measurement features for t-SNE.
3. Fits `TSNE(learning_rate=50)` on `df_num` and stores the resulting 2D coordinates as `x`/`y` columns.
4. Plots the embedding as a scatterplot with no coloring, then re-plots it colored (`hue`) by individual numeric features (e.g. `bicepscircumferenceflexed`) and by `BMI_class` pulled back in from the original `df`.

## Takeaway

The uncolored embedding forms one large, undifferentiated cluster — there's no obvious separation into distinct body-shape groups. Coloring by individual measurements or by `BMI_class` shows those values vary smoothly across the embedding rather than aligning with any cluster boundaries. Coloring by `Height_class` does line up with the horizontal axis: tall people cluster to the right, small people to the left, normal-height in the middle — height is the main driver of variance the embedding picks up on.

## Feature selection (in progress)

Continuing on from the t-SNE result, the notebook now explores trimming `df_num` down with `VarianceThreshold`:

- A first pass at `threshold=1` on the raw features doesn't drop anything — the mask keeps all columns, since threshold=1 isn't meaningful on unnormalized measurements with wildly different scales.
- Normalizing by the column mean (`df_num / df_num.mean()`) before fitting, with a much smaller `threshold=0.005`, drops the feature count from 95 to 29 — this is the useful pass, since normalization puts variance on a comparable scale across features before thresholding.
- The resulting 29-feature `reduced_df` is then checked for multicollinearity: a correlation matrix (`.corr()`), visualized as a heatmap (masking the upper triangle to avoid showing duplicate/self correlations), and a separate absolute-value correlation matrix (`tri_df`) with the upper triangle masked out, as a first step toward identifying and dropping highly correlated feature pairs.

This part of the notebook is still being worked on — the next step (not yet written) is presumably picking a correlation cutoff and dropping one feature from each highly-correlated pair.