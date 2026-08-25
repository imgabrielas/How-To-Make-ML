"""
Animated NMF reconstruction visualization for a single face image.

Shows how a chosen image from the Olivetti faces dataset is rebuilt as a
weighted sum of NMF learned "parts", added one at a time in order of
contribution (largest weight first). Ends with a gallery of every learned
component.

Requirements:
    pip install numpy matplotlib scikit-learn scipy pillow

Dataset:
    olivettifaces.mat

Outputs:
    figures/nmf_reconstruction.gif
    figures/nmf_components_gallery.png
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.io import loadmat
from sklearn.decomposition import NMF

RNG_SEED = 0
N_COMPONENTS = 25
IMAGE_SHAPE = (64, 64)
HOLD_FRAMES = 5

CMAP_RECON = "gray"
CMAP_PART = "viridis"
COLOR_BAR_USED = "#7648f4"
COLOR_BAR_UNUSED = "#e6e6e6"

HERE = os.path.dirname(__file__)


def load_data():
    data = loadmat(os.path.join(HERE, "olivettifaces.mat"))
    faces = data["faces"]
    return faces.T


def fit_nmf(X, n_components, seed):
    model = NMF(n_components=n_components, init="nndsvda", random_state=seed, max_iter=500)
    W = model.fit_transform(X)
    H = model.components_
    return W, H


def build_animation(image_index, X, W, H, out_path):
    image = X[image_index].reshape(IMAGE_SHAPE)
    weights = W[image_index]
    order = np.argsort(weights)[::-1]
    n_components = len(order)

    full_recon = (weights @ H).reshape(IMAGE_SHAPE)
    vmax = max(image.max(), full_recon.max())

    fig = plt.figure(figsize=(13, 5))
    ax_orig = fig.add_subplot(141)
    ax_recon = fig.add_subplot(142)
    ax_part = fig.add_subplot(143)
    ax_bar = fig.add_subplot(144)

    n_frames = n_components + HOLD_FRAMES

    def update(frame):
        step = min(frame, n_components)
        used = order[:step]

        cumulative = (weights[used] @ H[used]).reshape(IMAGE_SHAPE) if step else np.zeros(IMAGE_SHAPE)

        ax_orig.cla()
        ax_orig.imshow(image, cmap=CMAP_RECON, vmin=0, vmax=vmax)
        ax_orig.set_title("Original")
        ax_orig.axis("off")

        ax_recon.cla()
        ax_recon.imshow(cumulative, cmap=CMAP_RECON, vmin=0, vmax=vmax)
        ax_recon.set_title(f"Reconstruction\n{step}/{n_components} parts")
        ax_recon.axis("off")

        ax_part.cla()
        if step < n_components:
            comp_idx = order[step]
            part = H[comp_idx].reshape(IMAGE_SHAPE)
            ax_part.imshow(part, cmap=CMAP_PART)
            ax_part.set_title(f"+ Part #{comp_idx}\nweight={weights[comp_idx]:.3f}")
        else:
            ax_part.imshow(np.zeros(IMAGE_SHAPE), cmap=CMAP_PART)
            ax_part.set_title("Done")
        ax_part.axis("off")

        ax_bar.cla()
        colors = [COLOR_BAR_USED if i in used else COLOR_BAR_UNUSED for i in range(n_components)]
        ax_bar.bar(range(n_components), weights, color=colors)
        ax_bar.set_title("Part weights used")
        ax_bar.set_xlabel("Component #")
        ax_bar.set_ylabel("Weight")
        ax_bar.set_xlim(-1, n_components)

        fig.suptitle(f"NMF Reconstruction — face #{image_index}", fontsize=14)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=400, repeat=True)
    anim.save(out_path, writer=PillowWriter(fps=3))
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_components_gallery(H, out_path):
    n_components = H.shape[0]
    n_cols = int(np.ceil(np.sqrt(n_components)))
    n_rows = int(np.ceil(n_components / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

    for i, ax in enumerate(axes.flat):
        if i < n_components:
            ax.imshow(H[i].reshape(IMAGE_SHAPE), cmap=CMAP_PART)
        ax.axis("off")

    fig.suptitle("NMF Learned Components (Parts)", fontsize=16)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-index", type=int, default=0, help="Index of the face to reconstruct (0-399)")
    parser.add_argument("--n-components", type=int, default=N_COMPONENTS, help="Number of NMF components to learn")
    args = parser.parse_args()

    figures_dir = os.path.join(HERE, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    X = load_data()
    W, H = fit_nmf(X, args.n_components, RNG_SEED)

    build_animation(
        args.image_index,
        X,
        W,
        H,
        os.path.join(figures_dir, "nmf_reconstruction.gif"),
    )

    plot_components_gallery(H, os.path.join(figures_dir, "nmf_components_gallery.png"))


if __name__ == "__main__":
    main()