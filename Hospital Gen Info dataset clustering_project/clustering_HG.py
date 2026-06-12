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
INPUT_FILE = "input/hospital general info medians.xlsx"
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

# ================= HARDCODED SHORT NAMES =================
COLUMN_SHORT_NAMES = {
    "Average Length of Stay": "AvgLOS",
    "Emergency Department Volume": "EDVol",
    "Hospital Overall Rating": "OverallRt",
    "Mortality Rate": "MortRate",
    "Readmission Rate": "ReadmitRt",
    "Patient Experience Score": "PtExp",
    "Safety of Care Score": "Safety",
    "Effectiveness of Care Score": "Effect",
    "Timeliness of Care Score": "Timely",
    "Efficient Use of Medical Imaging": "Imaging",
    "Complication Rate": "CompRate",
    "Infection Rate": "InfectRt",
}
# =========================================================

os.makedirs(f"{OUTPUT_DIR}/dendrograms", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/kmeans", exist_ok=True)

# ================= LOAD DATA =================
df = pd.read_excel(INPUT_FILE)

# First column = entity identifier (state / hospital / facility)
entity_labels = df.iloc[:, 0].astype(str)

# Numeric columns only
numeric_df = df.select_dtypes(include=[np.number])

# Apply hardcoded short names where available
numeric_df.rename(columns=COLUMN_SHORT_NAMES, inplace=True)

# ================= SCALING =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)
X_cosine = normalize(X_scaled)

all_k_values = set()

# ================= DENDROGRAM (AUTO CLUSTERS) =================
def save_dendrogram_auto(Z, title, filename):
    max_dist = np.max(Z[:, 2])
    threshold = 0.7 * max_dist  # data-driven cut

    clusters = fcluster(Z, threshold, criterion="distance")
    k = len(np.unique(clusters))
    all_k_values.add(k)

    plt.figure(figsize=(14, 6))
    dendrogram(Z, labels=entity_labels.values, leaf_rotation=90)

    # Cluster -> entity mapping
    cluster_map = {}
    for entity, cid in zip(entity_labels, clusters):
        cluster_map.setdefault(cid, []).append(entity)

    colors = plt.cm.tab10.colors
    legend_elements = []

    for idx, (cid, members) in enumerate(cluster_map.items()):
        legend_elements.append(
            Line2D(
                [0], [0],
                color=colors[idx % len(colors)],
                lw=2,
                label=f"Cluster {idx + 1} ({', '.join(members)})"
            )
        )

    plt.legend(handles=legend_elements, loc="upper right", fontsize=9)
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

    # Cluster -> entity mapping
    cluster_entities = {}
    for i in range(k):
        cluster_entities[i] = entity_labels[labels_km == i].tolist()

    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=numeric_df.columns
    )

    plt.figure(figsize=(14, 6))
    for i in range(k):
        plt.plot(
            centroids.columns,
            centroids.iloc[i],
            marker="o",
            label=f"Cluster {i + 1} ({', '.join(cluster_entities[i])})"
        )

    plt.xticks(rotation=90)
    plt.ylabel("Centroid Value")
    plt.title(f"K-means Centroid Profile Plot (K = {k})")
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kmeans/kmeans_centroids_k{k}.png")
    plt.close()

print("✅ Auto-clustering completed successfully.")
