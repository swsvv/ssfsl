import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm.rich import tqdm

from util.get_module import get_encoder, get_transform
from util.set_seed import set_seed


@torch.no_grad()
def test(args, ckpt: str) -> None:
    ckpt_path = Path(args.exp_path, ckpt)

    transform = get_transform(args.dataset, "EpisodeTest")

    full_dataset = ImageFolder(
        root=Path(args.dataset_path, "train"), transform=transform
    )

    # Select 5 classes uniformly
    all_classes = full_dataset.classes
    selected_classes = np.random.choice(all_classes, args.num_classes, replace=False)

    # Get class to index mapping
    class_to_idx = full_dataset.class_to_idx
    selected_class_indices = [class_to_idx[c] for c in selected_classes]

    # Create a map from class index to a list of sample indices
    class_to_indices = {i: [] for i in range(len(all_classes))}
    for i, (_, label) in enumerate(full_dataset.samples):
        class_to_indices[label].append(i)

    # Get indices for the selected classes and samples
    indices = []
    for class_idx in selected_class_indices:
        class_indices = class_to_indices[class_idx]
        selected_indices = np.random.choice(
            class_indices, args.num_samples, replace=False
        )
        indices.extend(selected_indices)

    dataset = Subset(full_dataset, indices)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,  # Do not drop last batch
    )

    encoder = get_encoder(args.encoder)
    encoder = encoder.to(args.device)
    encoder.eval()

    ckpt_state_dict = torch.load(ckpt_path, map_location=args.device)
    encoder.load_state_dict(ckpt_state_dict["encoder_state_dict"])

    all_embeddings = []
    all_labels = []

    for idx, (x, y) in enumerate(tqdm(data_loader)):
        x = x.to(args.device)
        y = y.to(args.device)

        embeddings = encoder(x)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(y.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    viz(all_embeddings, all_labels, args)

    # if args.save_embeddings_path:
    #     filename_base = f"{args.model_name}_c{args.num_classes}_s{args.num_samples}_p{args.tsne_perplexity}_i{args.tsne_n_iter}_lr{args.tsne_learning_rate}"
    #     save_path = Path(args.save_embeddings_path, filename_base)
    #     save_path.parent.mkdir(parents=True, exist_ok=True)
    #     torch.save(
    #         {"embeddings": all_embeddings, "labels": all_labels},
    #         save_path,
    #     )


def viz(embeddings: Tensor, labels: Tensor, args) -> None:
    # Convert tensors to numpy arrays
    embeddings_np = embeddings.numpy()
    labels_np = labels.numpy()

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_np)

    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings_scaled)

    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        n_iter=args.tsne_n_iter,
        learning_rate=args.tsne_learning_rate,
        init="pca",
        random_state=args.seed,
    )

    embeddings_tsne = tsne.fit_transform(embeddings_pca)

    unique_labels = np.unique(labels_np)
    label_mapping = {label: f"Class {i}" for i, label in enumerate(unique_labels)}
    mapped_labels = np.array([label_mapping[label] for label in labels_np])
    hue_order = [f"Class {i}" for i in range(len(unique_labels))]

    if args.num_classes <= 10:
        palette = "tab10"
    elif args.num_classes <= 20:
        palette = "tab20"
    else:
        palette = sns.color_palette("hsv", args.num_classes)

    plt.figure(figsize=(10, 8))
    ax = sns.scatterplot(
        x=embeddings_tsne[:, 0],
        y=embeddings_tsne[:, 1],
        hue=mapped_labels,
        hue_order=hue_order,
        palette=palette,
        legend=False,
        alpha=0.8,
    )
    plt.xticks([])
    plt.yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Create the save directory if it doesn't exist
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the plot
    filename_base = f"{args.model_name}_c{args.num_classes}_s{args.num_samples}_p{args.tsne_perplexity}_i{args.tsne_n_iter}_lr{args.tsne_learning_rate}"
    plt.savefig(save_dir / f"tsne_{filename_base}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize embeddings using t-SNE")
    parser.add_argument(
        "--exp_path", type=str, required=True, help="Path to the experiment directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="miniImageNet",
        help="Name of the dataset",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
        default="/path/to/dataset/miniImageNet/Ravi",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for data loading"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        help="Name of the encoder model",
        default="ResNet18",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for computation (e.g., 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility", default=42
    )
    parser.add_argument(
        "--ckpt", type=str, default="ckpt-499.pkl", help="Name of the checkpoint file"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./img",
        help="Directory to save the t-SNE plot image",
    )
    parser.add_argument(
        "--save_embeddings_path",
        type=str,
        default="img",
        help="File path to save the generated embeddings",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name to visualize",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="Number of classes to visualize",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of samples per class to visualize",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=int,
        default=30,
        help="Perplexity for t-SNE",
    )
    parser.add_argument(
        "--tsne_n_iter",
        type=int,
        default=1000,
        help="Number of iterations for t-SNE",
    )
    parser.add_argument(
        "--tsne_learning_rate",
        type=int,
        default=200,
        help="Learning rate for t-SNE",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    test(args, args.ckpt)
