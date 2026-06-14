# ============================================================
# Yelp Cluster-wise Emotion Analysis
# Generates:
#   1) Emotion Count Bar Plot
#   2) Emotion Percentage Horizontal Plot
# For EACH cluster
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from nrclex import NRCLex

# ================= CONFIG =================

INPUT_FILE = "input/yelp_reviews.xlsx"
TEXT_COL = "text"
STATE_COL = "state"

OUTPUT_DIR = "output/cluster_emotion_plots"
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

# ================= EMOTIONS =================

EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust"
]

# ================= COLOR MAP =================
# Negative → red family
# Neutral → yellow / purple
# Positive → green / blue

EMOTION_COLORS = {
    "anger": "darkred",
    "disgust": "crimson",
    "fear": "firebrick",
    "sadness": "maroon",
    "anticipation": "goldenrod",
    "surprise": "purple",
    "joy": "seagreen",
    "trust": "steelblue"
}

# ================= LOAD DATA =================

df = pd.read_excel(INPUT_FILE)
df[TEXT_COL] = df[TEXT_COL].astype(str)
df[STATE_COL] = df[STATE_COL].astype(str)

print("Yelp dataset loaded successfully.")

# ================= PROCESS CLUSTERS =================

for cluster_name, states in CLUSTERS.items():

    print(f"\nProcessing: {cluster_name}")

    cluster_df = df[df[STATE_COL].isin(states)].copy()

    if cluster_df.empty:
        print("No data for this cluster.")
        continue

    # Aggregate emotion counts
    emotion_totals = {e: 0 for e in EMOTIONS}

    for text in cluster_df[TEXT_COL]:
        emotion = NRCLex(text)
        scores = emotion.raw_emotion_scores

        for e in EMOTIONS:
            emotion_totals[e] += scores.get(e, 0)

    total_emotions = sum(emotion_totals.values())

    if total_emotions == 0:
        print("No emotion words found.")
        continue

    counts = [emotion_totals[e] for e in EMOTIONS]
    percentages = [emotion_totals[e] / total_emotions for e in EMOTIONS]

    colors = [EMOTION_COLORS[e] for e in EMOTIONS]

    # ============================================================
    # PLOT 1: EMOTION COUNTS
    # ============================================================

    plt.figure(figsize=(9, 6))
    plt.bar(EMOTIONS, counts, color=colors)
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    plt.title(f"{cluster_name} — Emotion Counts")
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{cluster_name}_emotion_counts.png",
        dpi=300
    )
    plt.close()

    # ============================================================
    # PLOT 2: EMOTION PERCENTAGES (SORTED)
    # ============================================================

    sorted_pairs = sorted(
        zip(EMOTIONS, percentages),
        key=lambda x: x[1],
        reverse=True
    )

    emotions_sorted, perc_sorted = zip(*sorted_pairs)
    sorted_colors = [EMOTION_COLORS[e] for e in emotions_sorted]

    plt.figure(figsize=(9, 6))
    plt.barh(emotions_sorted, perc_sorted, color=sorted_colors)
    plt.xlabel("Percentage")
    plt.title(f"{cluster_name} — Emotion Distribution")
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{cluster_name}_emotion_percentages.png",
        dpi=300
    )
    plt.close()

    print("Plots saved.")

print("\nAll cluster emotion plots generated successfully.")