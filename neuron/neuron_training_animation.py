"""
Animated training visualization for a single-neuron classifier.

Requirements:
    pip install numpy pandas matplotlib scikit-learn pillow

Dataset:
    neuron_dataset.csv (semicolon separated)

Outputs:
    neuron_training.gif
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.model_selection import train_test_split

RNG_SEED = 42
MAX_ITERATIONS = 10
LEARNING_RATE = 1.0

COLOR_CLASS_1 = "#7648f4"
COLOR_CLASS_0 = "#ef77a7"
COLOR_BOUNDARY = "#fbedc0"


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


def predict(X, w, b):
    return (predict_proba(X, w, b) >= 0.5).astype(int)


def bce_loss(y, p):
    eps = 1e-12
    p = np.clip(p, eps, 1-eps)
    return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))


def to_raw_space(w, b, mu, sigma):
    w_raw = w / sigma
    b_raw = b - np.sum(w * mu / sigma)
    return w_raw, b_raw


def train_neuron(X, y, w, b, iterations, lr):
    history = []

    for epoch in range(iterations + 1):
        proba = predict_proba(X, w, b)
        loss = bce_loss(y, proba)

        history.append({
            "w": w.copy(),
            "b": b,
            "loss": loss
        })

        if epoch == iterations:
            break

        error = proba - y
        grad_w = (X.T @ error) / len(X)
        grad_b = np.mean(error)

        w -= lr * grad_w
        b -= lr * grad_b

    return history


df = pd.read_csv(os.path.join(os.path.dirname(__file__), "neuron_dataset.csv"), sep=";")

X = df[["u1","u2","u3"]].values.astype(float)
y = df["label"].values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=RNG_SEED,
    stratify=y
)

mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)

X_train_s = (X_train-mu)/sigma
X_test_s = (X_test-mu)/sigma

rng = np.random.default_rng(RNG_SEED)
w = rng.normal(scale=0.1, size=3)
b = 0.0

history = train_neuron(
    X_train_s,
    y_train,
    w,
    b,
    MAX_ITERATIONS,
    LEARNING_RATE
)

fig = plt.figure(figsize=(12,6))

ax3d = fig.add_subplot(121, projection="3d")
axloss = fig.add_subplot(222)
axtext = fig.add_subplot(224)

mask1 = y_train == 1
mask0 = y_train == 0


def update(frame):

    ax3d.cla()
    axloss.cla()
    axtext.cla()

    h = history[frame]

    w_raw, b_raw = to_raw_space(h["w"], h["b"], mu, sigma)

    ax3d.scatter(
        X_train[mask1,0],
        X_train[mask1,1],
        X_train[mask1,2],
        c=COLOR_CLASS_1,
        s=60,
        label="Class 1"
    )

    ax3d.scatter(
        X_train[mask0,0],
        X_train[mask0,1],
        X_train[mask0,2],
        c=COLOR_CLASS_0,
        marker="^",
        s=60,
        label="Class 0"
    )

    if abs(w_raw[2]) > 1e-8:

        u1 = np.linspace(X_train[:,0].min()-1,
                         X_train[:,0].max()+1,10)

        u2 = np.linspace(X_train[:,1].min()-1,
                         X_train[:,1].max()+1,10)

        U1,U2 = np.meshgrid(u1,u2)

        U3 = -(w_raw[0]*U1 + w_raw[1]*U2 + b_raw)/w_raw[2]

        ax3d.plot_surface(
            U1,
            U2,
            U3,
            color=COLOR_BOUNDARY,
            alpha=0.35
        )

    ax3d.set_title(f"Decision Boundary\nEpoch {frame}")
    ax3d.set_xlabel("u1")
    ax3d.set_ylabel("u2")
    ax3d.set_zlabel("u3")
    ax3d.legend()

    losses = [x["loss"] for x in history[:frame+1]]

    axloss.plot(range(frame+1), losses, marker="o")
    axloss.set_xlim(0, MAX_ITERATIONS)
    axloss.set_ylim(min([x["loss"] for x in history])-0.05,
                    max([x["loss"] for x in history])+0.05)
    axloss.set_title("Binary Cross Entropy Loss")
    axloss.set_xlabel("Epoch")
    axloss.set_ylabel("Loss")
    axloss.grid(True)

    pred = predict(X_test_s, h["w"], h["b"])
    acc = np.mean(pred == y_test)

    axtext.axis("off")

    txt = (
        f"Epoch: {frame}/{MAX_ITERATIONS}\n\n"
        f"Loss: {h['loss']:.4f}\n\n"
        f"Accuracy: {acc:.2f}\n\n"
        f"w1 = {h['w'][0]:.3f}\n"
        f"w2 = {h['w'][1]:.3f}\n"
        f"w3 = {h['w'][2]:.3f}\n\n"
        f"bias = {h['b']:.3f}"
    )

    axtext.text(
        0.02,
        0.98,
        txt,
        fontsize=12,
        va="top",
        family="monospace"
    )

    fig.suptitle("Single Neuron Training Animation", fontsize=16)


anim = FuncAnimation(
    fig,
    update,
    frames=len(history),
    interval=900,
    repeat=True
)

anim.save(
    "neuron_training.gif",
    writer=PillowWriter(fps=1)
)

print("Saved neuron_training.gif")

plt.show()
