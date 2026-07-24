# NMF (Non-negative Matrix Factorization)

An exercise in decomposing image datasets into non-negative components using scikit-learn's `NMF`.

## Data

- [mnist.npz](mnist.npz) — the standard MNIST handwritten digit dataset (28x28 grayscale images), loaded with `np.load` into `X_train`/`y_train`/`X_test`/`y_test`.
- [olivettifaces.mat](olivettifaces.mat) — the AT&T/Olivetti Faces dataset, a classic NMF example (Lee & Seung) of grayscale face images. It's a MATLAB 5.0 `.mat` file, so it needs `scipy.io.loadmat` to open — not `pickle` or plain `np.load`.

## What it does

[example.ipynb](example.ipynb) — reference example:

1. Loads MNIST digits from `mnist.npz`.
2. Fits an `NMF` model with 8 components on digit image data.
3. Reshapes and visualizes the learned components as images.
4. Repeats on a larger 100-image sample flattened to 784-length vectors, and visualizes the resulting components.

[notebook.ipynb](notebook.ipynb) — the exercise notebook, applied NMF to the Olivetti faces dataset, similar to exercise.
