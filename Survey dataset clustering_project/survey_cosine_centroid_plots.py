import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler, normalize

# ================= CONFIG =================
INPUT_FILE = "input/Survey Data Median.xlsx"
OUTPUT_DIR = "output/centroids"

METHODS = [
    ("complete", "cosine"),
    ("average", "cosine"),
]
# =========================================

# ================= SHORT LABELS =================
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
# ===============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD DATA =================
df = pd.read_excel(INPUT_FILE)

state_labels = df.iloc[:, 0].astype(str)

numeric_df = df.select_dtypes(include=[np.number])
numeric_df.rename(columns=SHORT_LABELS, inplace=True)

# ================= SCALING =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)
X_cosine = normalize(X_scaled)

# ================= CORE FUNCTION =================
def hierarchical_cosine_centroid_plot(method, title, output_file):

    dist_matrix = pdist(X_cosine, metric="cosine")
    Z = linkage(dist_matrix, method=method)

    max_dist = np.max(Z[:, 2])
    threshold = 0.7 * max_dist
    clusters = fcluster(Z, threshold, criterion="distance")

    unique_clusters = np.unique(clusters)

    centroids = []
    cluster_states = {}

    for cid in unique_clusters:
        members = X_cosine[clusters == cid]
        centroids.append(members.mean(axis=0))
        cluster_states[cid] = state_labels[clusters == cid].tolist()

    centroids = np.array(centroids)

    # ================= PLOT =================
    plt.figure(figsize=(14, 6))

    for idx, centroid in enumerate(centroids):
        plt.plot(
            numeric_df.columns,
            centroid,
            marker="o",
            label=f"Cluster {idx + 1} ({', '.join(cluster_states[unique_clusters[idx]])})"
        )

    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.ylabel("Normalized Centroid Value")
    plt.title(f"{title} | Auto clusters = {len(unique_clusters)}")
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_file)
    plt.close()

# ================= RUN BOTH CASES =================
hierarchical_cosine_centroid_plot(
    method="complete",
    title="Hierarchical Centroid Plot (Cosine + Complete)",
    output_file=f"{OUTPUT_DIR}/survey_cosine_complete_centroid.png"
)

hierarchical_cosine_centroid_plot(
    method="average",
    title="Hierarchical Centroid Plot (Cosine + Average)",
    output_file=f"{OUTPUT_DIR}/survey_cosine_average_centroid.png"
)

print("✅ Survey cosine centroid plots generated successfully.")
