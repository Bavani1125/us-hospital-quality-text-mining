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

os.makedirs(f"{OUTPUT_DIR}/dendrograms", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/kmeans", exist_ok=True)

# ================= LOAD DATA =================
df = pd.read_excel(INPUT_FILE)

entity_labels = df.iloc[:, 0].astype(str)
numeric_df = df.select_dtypes(include=[np.number])
numeric_df.rename(columns=COLUMN_SHORT_NAMES, inplace=True)

# ================= SCALING =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)
X_cosine = normalize(X_scaled)

all_k_values = set()

# ================= DENDROGRAM (AUTO CLUSTERS) =================
def save_dendrogram_auto(Z, title, filename):
    max_dist = np.max(Z[:, 2])
    threshold = 0.7 * max_dist

    clusters = fcluster(Z, threshold, criterion="distance")
    k = len(np.unique(clusters))
    all_k_values.add(k)

    # UPDATED FIGURE SIZE (more height)
    plt.figure(figsize=(10, 8))

    dendrogram(
        Z,
        labels=entity_labels.values,
        leaf_rotation=90,
        leaf_font_size=12
    )

    # Constrict X-axis spacing
    plt.gca().margins(x=0.01)

    # Cluster mapping
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
                lw=3,
                label=f"Cluster {idx + 1} ({', '.join(members)})"
            )
        )

    # UPDATED LEGEND
    plt.legend(
    handles=legend_elements,
    loc="upper right",
    fontsize=18,          # ↑ bigger text
    markerscale=1.8,      # ↑ bigger legend lines
    handlelength=3,       # ↑ longer legend lines
    borderpad=1.2,        # ↑ spacing inside box
    labelspacing=1.2,     # ↑ space between entries
    frameon=True
    )

    # UPDATED TITLE SIZE
    plt.title(f"{title} | Auto clusters = {k}", fontsize=16)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ================= RUN DENDROGRAMS =================
for method, distance in SET_A + SET_B:

    if method == "ward" and distance != "euclidean":
        continue

    data = X_scaled if distance == "euclidean" else X_cosine
    dist_matrix = pdist(data, metric=distance)
    Z = linkage(dist_matrix, method=method)

    save_dendrogram_auto(
        Z,
        title=f"{method.upper()} linkage ({distance.upper()})",
        filename=f"{OUTPUT_DIR}/dendrograms/{method}_{distance}.png"
    )

# ================= K-MEANS =================
for k in sorted(all_k_values):

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(X_scaled)

    cluster_entities = {}
    for i in range(k):
        cluster_entities[i] = entity_labels[labels_km == i].tolist()

    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=numeric_df.columns
    )

    plt.figure(figsize=(10, 6))

    for i in range(k):
        plt.plot(
            centroids.columns,
            centroids.iloc[i],
            marker="o",
            label=f"Cluster {i + 1} ({', '.join(cluster_entities[i])})"
        )

    plt.xticks(rotation=90)
    plt.ylabel("Centroid Value")
    plt.title(f"K-means Centroid Profile Plot (K = {k})", fontsize=14)

    plt.legend(loc="upper right", fontsize=10)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kmeans/kmeans_centroids_k{k}.png")
    plt.close()

print("✅ Auto-clustering completed successfully.")