# ============================================================
# Passion Index Analysis (Cluster-wise)
# Updated Version:
# - 4 quadrant division only
# - Mean-based crosshair placement (paper-inspired logic)
# - Balanced bubble scaling
# - Clean visual labeling
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import numpy as np

from nltk.sentiment import SentimentIntensityAnalyzer

# ================= NLTK SETUP =================
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# ================= CONFIG =================
INPUT_FILE = "input/yelp_reviews.xlsx"
OUTPUT_DIR = "output/passion_index_mean_crosshair"

TEXT_COL = "text"
STATE_COL = "state"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= CLUSTERS =================
CLUSTERS = {
    "HG-C1": ["AZ", "FL", "NJ", "IL"],
    "HG-C2": ["ID", "TN"],
    "HG-C3": ["IN", "PA", "MO", "CA"],

    "S-C1": ["ID", "IL", "LA", "PA", "TN"],
    "S-C2": ["AZ", "IN", "NJ"],
    "S-C3": ["FL", "MO"],
}

# ================= COLOR MAP =================
# Blue shades → Hospital General clusters
# Orange shades → Survey clusters
CLUSTER_COLORS = {
    "HG-C1": "#1f77b4",
    "HG-C2": "#6baed6",
    "HG-C3": "#bdd7e7",

    "S-C1": "#d94801",
    "S-C2": "#fd8d3c",
    "S-C3": "#fdd0a2",
}

# ================= LOAD DATA =================
df = pd.read_excel(INPUT_FILE)
df[TEXT_COL] = df[TEXT_COL].astype(str)
df[STATE_COL] = df[STATE_COL].astype(str)

# ================= SENTIMENT FUNCTION =================
def sentiment_score(text):
    return sia.polarity_scores(text)["compound"]

# ================= MAIN ANALYSIS =================
results = []

for cluster_name, states in CLUSTERS.items():
    cluster_df = df[df[STATE_COL].isin(states)].copy()

    if cluster_df.empty:
        continue

    cluster_df["compound"] = cluster_df[TEXT_COL].apply(sentiment_score)

    results.append({
        "cluster": cluster_name,
        "polarity": cluster_df["compound"].mean(),          # Y-axis
        "intensity": cluster_df["compound"].abs().mean(),   # X-axis
        "volume": len(cluster_df)
    })

result_df = pd.DataFrame(results)

if result_df.empty:
    raise ValueError("No data available after filtering. Please check input file and cluster states.")

# ================= BUBBLE SIZE SCALING =================
volumes = result_df["volume"].values
sizes = np.sqrt(volumes) * 120

# ================= AXIS CALIBRATION =================
# Plot limits still use min/max + padding for display
x_min, x_max = result_df["intensity"].min(), result_df["intensity"].max()
y_min, y_max = result_df["polarity"].min(), result_df["polarity"].max()

x_pad = (x_max - x_min) * 0.25 if x_max != x_min else 0.05
y_pad = (y_max - y_min) * 0.25 if y_max != y_min else 0.05

# ================= MEAN-BASED CROSSHAIR LOGIC =================
# This is the key methodological update:
# crosshairs are placed at the mean of plotted cluster values,
# not at the midpoint of the observed min-max range.
x_cross = result_df["intensity"].mean()
y_cross = result_df["polarity"].mean()

# ================= PLOTTING =================
plt.figure(figsize=(11, 8))

# Plot bubbles
for i, row in result_df.iterrows():
    plt.scatter(
        row["intensity"],
        row["polarity"],
        s=sizes[i],
        color=CLUSTER_COLORS.get(row["cluster"], "#999999"),
        alpha=0.75,
        edgecolors="black"
    )
    plt.text(
        row["intensity"],
        row["polarity"],
        row["cluster"],
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

# ================= 4-QUADRANT CROSSHAIRS =================
plt.axvline(
    x=x_cross,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    alpha=0.85
)

plt.axhline(
    y=y_cross,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    alpha=0.85
)

# ================= AXIS LIMITS =================
plt.xlim(x_min - x_pad, x_max + x_pad)
plt.ylim(y_min - y_pad, y_max + y_pad)

# ================= QUADRANT LABELS =================
plt.text(
    0.02, 0.95, "LIKE",
    transform=plt.gca().transAxes,
    fontsize=13,
    color="green",
    fontweight="bold"
)

plt.text(
    0.86, 0.95, "LOVE",
    transform=plt.gca().transAxes,
    fontsize=13,
    color="darkgreen",
    fontweight="bold"
)

plt.text(
    0.02, 0.05, "DISLIKE",
    transform=plt.gca().transAxes,
    fontsize=13,
    color="orange",
    fontweight="bold"
)

plt.text(
    0.86, 0.05, "HATE",
    transform=plt.gca().transAxes,
    fontsize=13,
    color="red",
    fontweight="bold"
)

# ================= FINAL LABELS =================
plt.xlabel("Passion Intensity (Average |Sentiment|)")
plt.ylabel("Sentiment Polarity (Average Sentiment)")
plt.title("Passion Index — Cluster-wise Sentiment Positioning (Mean-Based Crosshairs)")

plt.tight_layout()

output_path = f"{OUTPUT_DIR}/cluster_passion_index_mean_crosshair.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"✅ Passion Index plot saved: {output_path}")
print("🎉 Passion Index analysis completed successfully.")
print(f"📍 Vertical crosshair (Intensity mean): {x_cross:.4f}")
print(f"📍 Horizontal crosshair (Polarity mean): {y_cross:.4f}")