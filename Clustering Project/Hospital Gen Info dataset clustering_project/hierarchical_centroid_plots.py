import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler, normalize
from matplotlib.lines import Line2D

# ================= CONFIG =================
INPUT_FILE = "input/hospital general info medians.xlsx"
OUTPUT_DIR = "output/centroids"

METHOD = "complete"
EUCLIDEAN = "euclidean"
COSINE = "cosine"

# =========================================

# ================= HARD-CODED SHORT NAMES =================
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
# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD DATA =================
df = pd.read_excel(INPUT_FILE)

# First column = entity label (state / hospital / facility)
entity_labels = df.iloc[:, 0].astype(str)

# Numeric columns only
numeric_df = df.select_dtypes(include=[np.number])
numeric_df.rename(columns=COLUMN_SHORT_NAMES, inplace=True)

# ================= SCALING =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)
X_cosine = normalize(X_scaled)

# ================= CORE FUNCTION =================
def hierarchical_centroid_plot(data, distance, title, output_file):

    dist_matrix = pdist(data, metric=distance)
    Z = linkage(dist_matrix, method=METHOD)

    # Same auto-cut logic as dendrograms
    max_dist = np.max(Z[:, 2])
    threshold = 0.7 * max_dist
    clusters = fcluster(Z, threshold, criterion="distance")

    unique_clusters = np.unique(clusters)

    # Compute centroids in standardized / normalized space
    centroids = []
    cluster_entities = {}

    for cid in unique_clusters:
        members = data[clusters == cid]
        centroids.append(members.mean(axis=0))
        cluster_entities[cid] = entity_labels[clusters == cid].tolist()

    centroids = np.array(centroids)

    # ================= PLOT =================
    plt.figure(figsize=(14, 6))

    for idx, centroid in enumerate(centroids):
        plt.plot(
            numeric_df.columns,
            centroid,
            marker="o",
            label=f"Cluster {idx + 1} ({', '.join(cluster_entities[unique_clusters[idx]])})"
        )

    plt.xticks(rotation=90)
    plt.ylabel("Standardized / Normalized Centroid Value")
    plt.title(f"{title} | Auto clusters = {len(unique_clusters)}")
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# ================= RUN BOTH CASES =================
hierarchical_centroid_plot(
    data=X_scaled,
    distance=EUCLIDEAN,
    title="Hierarchical Centroid Plot (Complete + Euclidean)",
    output_file=f"{OUTPUT_DIR}/euclidean_complete_centroid.png"
)

hierarchical_centroid_plot(
    data=X_cosine,
    distance=COSINE,
    title="Hierarchical Centroid Plot (Complete + Cosine)",
    output_file=f"{OUTPUT_DIR}/cosine_complete_centroid.png"
)

print("✅ Hierarchical centroid plots generated successfully.")
