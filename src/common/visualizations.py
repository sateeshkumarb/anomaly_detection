import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE


def distance_distribution(
    positive_distances: np.ndarray, negative_distances: np.ndarray
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.kdeplot(
        positive_distances,
        ax=axes[0],
        fill=True,
        color="steelblue",
        alpha=0.5,
        label="Positive (normal)",
    )
    sns.kdeplot(
        negative_distances,
        ax=axes[0],
        fill=True,
        color="tomato",
        alpha=0.5,
        label="Negative (anomalous)",
    )

    # mark the means
    axes[0].axvline(
        positive_distances.mean(),
        color="steelblue",
        linestyle="--",
        linewidth=1.5,
        label=f"+ve mean: {positive_distances.mean():.3f}",
    )
    axes[0].axvline(
        negative_distances.mean(),
        color="tomato",
        linestyle="--",
        linewidth=1.5,
        label=f"-ve mean: {negative_distances.mean():.3f}",
    )

    axes[0].set_title("Distance Distribution from Anchor Centroid")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # --- Plot 2: Box plot (best for viewing spread and outliers) ---
    df = pd.DataFrame(
        {
            "distance": np.concatenate([positive_distances, negative_distances]),
            "type": ["Positive (normal)"] * len(positive_distances)
            + ["Negative (anomalous)"] * len(negative_distances),
        }
    )
    sns.boxplot(
        data=df,
        x="type",
        y="distance",
        hue="type",
        palette={"Positive (normal)": "steelblue", "Negative (anomalous)": "tomato"},
        showfliers=False,
        legend=False,
        ax=axes[1],
    )
    sns.stripplot(
        data=df,
        x="type",
        y="distance",
        hue="type",
        palette={"Positive (normal)": "steelblue", "Negative (anomalous)": "tomato"},
        alpha=0.4,
        size=4,
        jitter=True,
        legend=False,
        ax=axes[1],
    )  # overlay raw points

    # TODO: better understand this
    neg_outlier_threshold = np.percentile(
        positive_distances, 95
    )  # negatives below this are outliers
    pos_outlier_threshold = np.percentile(
        negative_distances, 10
    )  # positives above this are outliers

    n_neg_outliers = (negative_distances < neg_outlier_threshold).sum()
    n_pos_outliers = (positive_distances > pos_outlier_threshold).sum()

    for label, distances, x_pos, outlier_count in [
        ("Positive (normal)", positive_distances, 0, n_pos_outliers),
        ("Negative (anomalous)", negative_distances, 1, n_neg_outliers),
    ]:
        axes[1].text(
            x=x_pos,
            y=2.1,
            # s=f"n={len(distances)}\noutliers={outlier_count}",
            # TODO: include outlier count too in the plot
            s=f"n={len(distances)}",
            ha="center",
            fontsize=9,
        )

    axes[1].set_title("Distance Spread and Outliers")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Distance")

    plt.suptitle(
        "Distance Separation: Normal vs Anomalous",
        fontsize=20,
        fontweight="bold",
        y=0.95,
    )
    plt.tight_layout()
    plt.savefig("distance_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()


def tsne(embeddings: np.ndarray, labels: np.ndarray) -> None:
    tsne = TSNE(n_components=2, perplexity=20, verbose=1, random_state=42, n_jobs=4)
    reduced = tsne.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(10, 8))

    # https://matplotlib.org/stable/gallery/color/named_colors.html
    palette = {
        "Anchor (normal)": "forestgreen",
        "Positive (normal)": "mediumseagreen",
        "Negative (anomalous)": "magenta",
    }

    sns.scatterplot(
        x=reduced[:, 0],
        y=reduced[:, 1],
        hue=labels,
        palette=palette,
        alpha=0.6,
        s=20,  # point size
        ax=ax,
    )

    ax.set_title(
        "t-SNE Visualisation of Embedding Space\nCNN + Siamese Network",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(title="Sample Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
