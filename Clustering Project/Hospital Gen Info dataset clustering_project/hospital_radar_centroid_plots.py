import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler, normalize

# ================= CONFIG =================
INPUT_FILE = "input/hospital general info medians.xlsx"
OUTPUT_DIR = "output/radar_centroids"

METHOD = "complete"
CUT_RATIO = 0.7
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

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD DATA =================
df = pd.read_excel(INPUT_FILE)
entity_labels = df.iloc[:, 0].astype(str)

numeric_df = df.select_dtypes(include=[np.number])
numeric_df.rename(columns=COLUMN_SHORT_NAMES, inplace=True)

# ================= SCALING =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)
X_cosine = normalize(X_scaled)

# ================= RADAR FUNCTION =================
def radar_centroid_plot(data, distance, title, filename):

    dist_matrix = pdist(data, metric=distance)
    Z = linkage(dist_matrix, method=METHOD)

    threshold = CUT_RATIO * np.max(Z[:, 2])
    clusters = fcluster(Z, threshold, criterion="distance")
    unique_clusters = np.unique(clusters)

    centroids = []
    cluster_entities = {}

    for cid in unique_clusters:
        members = data[clusters == cid]
        centroids.append(members.mean(axis=0))
        cluster_entities[cid] = entity_labels[clusters == cid].tolist()

    centroids = np.array(centroids)

    labels = numeric_df.columns.tolist()
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for i, centroid in enumerate(centroids):
        values = centroid.tolist()
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=f"Cluster {i+1}")
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ================= RUN CASES =================
radar_centroid_plot(
    data=X_scaled,
    distance="euclidean",
    title="Radar Plot of Centroids (Complete + Euclidean)",
    filename=f"{OUTPUT_DIR}/radar_euclidean_complete.png"
)

radar_centroid_plot(
    data=X_cosine,
    distance="cosine",
    title="Radar Plot of Centroids (Complete + Cosine)",
    filename=f"{OUTPUT_DIR}/radar_cosine_complete.png"
)

print("✅ Radar centroid plots generated successfully.")
