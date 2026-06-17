import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from matplotlib.lines import Line2D

# ================= CONFIG =================
INPUT_FILE = "input/Survey Data Median.xlsx"
OUTPUT_DIR = "output"

SET_A = [
    ("single", "euclidean"),
    ("complete", "euclidean"),
    ("average", "euclidean"),
    ("ward", "euclidean"),
]

SET_B = [
    ("single", "cosine"),
    ("complete", "cosine"),
    ("average", "cosine"),
]
# =========================================

SHORT_LABELS = {
    "Patients who reported that staff definitely gave care in a professional way and the facility was clean": "Care-Def",
    "Patients who reported that staff somewhat gave care in a professional way or the facility was somewhat clean": "Care-Some",
    "Patients who reported that staff did not give care in a professional way or the facility was not clean": "Care-No",

    "Patients who reported that staff definitely communicated about what to expect during and after the procedure": "Comm-Def",
    "Patients who reported that staff somewhat communicated about what to expect during and after the procedure": "Comm-Some",
    "Patients who reported that staff did not communicate about what to expect during the procedure": "Comm-No",

    "Patients who gave the facility a rating of 9 or 10 on a scale from 0 (lowest) to 10 (highest)": "Rate-9_10",
    "Patients who gave the facility a rating of 7 or 8 on a scale from 0 (lowest) to 10 (highest)": "Rate-7_8",
    "Patients who gave the facility a rating of 0 to 6 on a scale from 0 (lowest) to 10 (highest)": "Rate-0_6",

    "Patients who reported YES they would DEFINITELY recommend the facility to family or friends": "Recom-Yes",
    "Patients who reported PROBABLY YES they would recommend the facility to family or friends": "Recom-Prob",
    "Patients who reported NO, they would not recommend the facility to family or friends": "Recom-No"
}

os.makedirs(f"{OUTPUT_DIR}/dendrograms", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/kmeans", exist_ok=True)

# ================= LOAD DATA =================
df = pd.read_excel(INPUT_FILE)
state_labels = df.iloc[:, 0].astype(str)

numeric_df = df.select_dtypes(include=[np.number])
numeric_df.rename(columns=SHORT_LABELS, inplace=True)

# ================= SCALING =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)
X_cosine = normalize(X_scaled)

all_k_values = set()

# ================= DENDROGRAM FUNCTION =================
def save_dendrogram_auto(Z, title, filename):
    max_dist = np.max(Z[:, 2])
    threshold = 0.7 * max_dist

    clusters = fcluster(Z, threshold, criterion="distance")
    k = len(np.unique(clusters))
    all_k_values.add(k)

    plt.figure(figsize=(7, 6))
    dendrogram(
        Z,
        labels=state_labels.values,
        leaf_rotation=90
    )

    cluster_map = {}
    for state, cid in zip(state_labels, clusters):
        cluster_map.setdefault(cid, []).append(state)

    colors = plt.cm.tab10.colors
    legend_elements = []

    for idx, (cid, states) in enumerate(cluster_map.items()):
        legend_elements.append(
            Line2D(
                [0], [0],
                color=colors[idx % len(colors)],
                lw=2,
                label=f"Cluster {idx + 1} ({', '.join(states)})"
            )
        )

    plt.legend(handles=legend_elements, loc="upper right", fontsize=13,
               framealpha=0.5)
    plt.title(f"{title} | Auto clusters = {k}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ================= SET A & SET B =================
for method, distance in SET_A + SET_B:

    if method == "ward" and distance != "euclidean":
        print(f"Skipping Ward + {distance}")
        continue

    data = X_scaled if distance == "euclidean" else X_cosine
    dist_matrix = pdist(data, metric=distance)
    Z = linkage(dist_matrix, method=method)

    save_dendrogram_auto(
        Z,
        title=f"{method.upper()} linkage ({distance.upper()})",
        filename=f"{OUTPUT_DIR}/dendrograms/{method}_{distance}.png"
    )

# ================= SET C: K-MEANS =================
for k in sorted(all_k_values):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(X_scaled)

    cluster_states = {}
    for i in range(k):
        cluster_states[i] = state_labels[labels_km == i].tolist()

    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=numeric_df.columns
    )

    plt.figure(figsize=(7, 6))
    for i in range(k):
        plt.plot(
            centroids.columns,
            centroids.iloc[i],
            marker="o",
            label=f"Cluster {i + 1} ({', '.join(cluster_states[i])})"
        )

    plt.xticks(rotation=90)
    plt.ylabel("Centroid Value")
    plt.title(f"K-means Centroid Profile Plot (K = {k})")
    plt.legend(loc="upper right", fontsize=13, framealpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kmeans/kmeans_centroids_k{k}.png")
    plt.close()

print("✅ Auto-clustering analysis completed successfully.")
