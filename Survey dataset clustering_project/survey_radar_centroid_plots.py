import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler, normalize

# ================= CONFIG =================
INPUT_FILE = "input/Survey Data Median.xlsx"
OUTPUT_DIR = "output/radar_centroids"
CUT_RATIO = 0.7

LINKAGE_METHODS = [
    ("average", "Cosine + Average"),
    ("complete", "Cosine + Complete"),
]
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD DATA =================
df = pd.read_excel(INPUT_FILE)

# First column = State
state_labels = df.iloc[:, 0].astype(str)

# Numeric survey columns (fixed order)
numeric_df = df.select_dtypes(include=[np.number])

# ================= CLEAN, POSITIONAL LABELS =================
numeric_df.columns = [
    "Care – High",
    "Care – Medium",
    "Care – Low",
    "Comm – High",
    "Comm – Medium",
    "Comm – Low",
    "Rating – High",
    "Rating – Medium",
    "Rating – Low",
    "Recommend – Yes",
    "Recommend – Maybe",
    "Recommend – No"
]

# ================= SCALING =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)
X_cosine = normalize(X_scaled)

# ================= RADAR FUNCTION =================
def radar_centroid_plot(linkage_method, title_suffix, output_name):

    # Hierarchical clustering
    dist_matrix = pdist(X_cosine, metric="cosine")
    Z = linkage(dist_matrix, method=linkage_method)

    threshold = CUT_RATIO * np.max(Z[:, 2])
    clusters = fcluster(Z, threshold, criterion="distance")
    unique_clusters = np.unique(clusters)

    # Centroids in cosine-normalized space
    centroids = []
    cluster_states = {}

    for cid in unique_clusters:
        members = X_cosine[clusters == cid]
        centroids.append(members.mean(axis=0))
        cluster_states[cid] = state_labels[clusters == cid].tolist()

    centroids = np.array(centroids)

    # Radar geometry
    labels = numeric_df.columns.tolist()
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # ================= PLOT =================
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for i, centroid in enumerate(centroids):
        values = centroid.tolist()
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=f"Cluster {i + 1}")
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(f"Survey Radar Plot of Centroids ({title_suffix})", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{output_name}")
    plt.close()

# ================= RUN BOTH =================
for method, title in LINKAGE_METHODS:
    radar_centroid_plot(
        linkage_method=method,
        title_suffix=title,
        output_name=f"survey_radar_{method}_cosine.png"
    )

print("✅ Survey radar centroid plots (Cosine + Average & Complete) generated successfully.")
