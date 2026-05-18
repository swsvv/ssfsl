from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm.rich import tqdm

from config import Config
from dataloader import get_dataloader
from util.ckpt import get_ckpt_list
from util.get_module import get_encoder, get_model
from util.set_seed import set_seed


@torch.no_grad()
def feat_viz(config: Config) -> None:
    ckpt_list = get_ckpt_list(config.exp_path)
    ckpt = ckpt_list[-1]
    ckpt_path = Path(config.exp_path, ckpt)

    test_loader = get_dataloader(config, "test")

    encoder = get_encoder(config.encoder)
    encoder = encoder.to(config.device)

    ckpt_state_dict = torch.load(ckpt_path, map_location=config.device)
    encoder.load_state_dict(ckpt_state_dict["encoder_state_dict"])

    model = get_model("EpisodeTest", config, encoder)
    model = model.to(config.device)
    model.eval()

    for episode_idx in tqdm(
        range(config.num_episode),
        desc=f"CKPT: {ckpt}, Test ({model.n_way}-way, {model.num_spt}-shot, {model.num_qry}-qry)",
        position=0,
        leave=False,
    ):
        for x, y in test_loader:
            x = x.to(config.device)
            y = y.to(config.device)

            feat = encoder(x)
            feat = feat.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            tsne_plot = visualize_features(feat, y, method="tsne")
            tsne_plot.savefig(f"tsne_{episode_idx}_{config.exp_name}.png")

            tsne_fig = visualize_features_3d(feat, y, method="tsne")
            tsne_fig.savefig(
                f"tsne_3d_{episode_idx}_{config.exp_name}.png",
                dpi=300,
                bbox_inches="tight",
            )


def visualize_features(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str] | None = None,
    method: str = "tsne",
    title: str | None = None,
    figsize=(10, 8),
):
    """
    Visualize features using dimensionality reduction techniques.

    Args:
        features: Extracted features (n_samples, n_features)
        labels: Corresponding labels (n_samples,)
        class_names: Names of the classes
        method: 'pca' or 'tsne'
        title: Title for the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Apply dimensionality reduction
    if method.lower() == "pca":
        reducer = PCA(n_components=2)
        reduced_features = reducer.fit_transform(features)
        method_name = "PCA"
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_features = reducer.fit_transform(features)
        method_name = "t-SNE"
    else:
        raise ValueError(f"Unknown method: {method}")

    # Plot
    unique_labels = np.unique(labels)
    if class_names is None:
        class_names = [f"Class {i}" for i in unique_labels]

    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(
            reduced_features[idx, 0],
            reduced_features[idx, 1],
            label=class_names[i] if i < len(class_names) else f"Class {label}",
            alpha=0.7,
        )

    plt.legend()
    plt.title(title if title else f"Feature visualization using {method_name}")
    plt.xlabel(f"{method_name} dimension 1")
    plt.ylabel(f"{method_name} dimension 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return plt


def visualize_features_3d(
    features,
    labels,
    class_names=None,
    method="pca",
    figsize=(12, 10),
    alpha=0.7,
    marker_size=50,
):
    """
    Visualize features in 3D space using different dimensionality reduction techniques.

    Args:
        features: Extracted features (n_samples, n_features)
        labels: Corresponding labels (n_samples,)
        class_names: Names of the classes
        method: 'pca', 'tsne', or 'raw' (if features are already 3D)
        figsize: Figure size tuple
        alpha: Transparency of points
        marker_size: Size of scatter points

    Returns:
        matplotlib figure object
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Apply dimensionality reduction to get 3D representation
    if method.lower() == "pca":
        reducer = PCA(n_components=3)
        reduced_features = reducer.fit_transform(features)
        method_name = "PCA"
        explained_var = sum(reducer.explained_variance_ratio_) * 100
        title = (
            f"{method_name} 3D projection (explains {explained_var:.1f}% of variance)"
        )
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=3, random_state=42)
        reduced_features = reducer.fit_transform(features)
        method_name = "t-SNE"
        title = f"{method_name} 3D projection"
    elif method.lower() == "raw":
        # Assume features are already 3D
        if features.shape[1] != 3:
            raise ValueError(
                "Features must have exactly 3 dimensions when method='raw'"
            )
        reduced_features = features
        method_name = "Raw"
        title = "3D feature space"
    else:
        raise ValueError(f"Unknown method: {method}")

    # Prepare for plotting
    unique_labels = np.unique(labels)
    if class_names is None:
        class_names = [f"Class {i}" for i in unique_labels]

    # Create a colormap - UPDATED from cm.get_cmap
    # In newer matplotlib versions, use colormaps[] instead of cm.get_cmap()
    cmap = colormaps["tab10"]

    # Plot each class
    for i, label in enumerate(unique_labels):
        idx = labels == label
        color = cmap(i % 10)  # Cycle through colors if more than 10 classes
        ax.scatter(
            reduced_features[idx, 0],
            reduced_features[idx, 1],
            reduced_features[idx, 2],
            s=marker_size,
            alpha=alpha,
            label=class_names[i] if i < len(class_names) else f"Class {label}",
            color=color,
        )

    # Add labels and legend
    ax.set_title(title)
    ax.set_xlabel(f"Dimension 1")
    ax.set_ylabel(f"Dimension 2")
    ax.set_zlabel(f"Dimension 3")

    # Add legend with smaller font size to avoid overlap
    ax.legend(loc="upper right", fontsize="small")

    # Improve visual appearance
    ax.grid(True, alpha=0.3)

    return fig


if __name__ == "__main__":
    config = Config.parse_args(require_default_file=True)

    if config.mode != "test":
        print(f"⚠️  Config's mode is {config.mode} -> change to 'test'!")
        config.mode = "test"

    # set_seed(config.seed)
    set_seed(42)

    feat_viz(config)
