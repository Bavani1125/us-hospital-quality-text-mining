# ============================================================
# Attribute Net Score Analysis (Cluster-wise)
# Clean label placement – no overlap with axis or ticks
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer

# ================= NLTK SETUP =================
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# ================= CONFIG =================
INPUT_FILE = "input/yelp_reviews.xlsx"
OUTPUT_DIR = "output/attribute_net_scores"

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

# ================= ATTRIBUTE KEYWORDS =================
ATTRIBUTES = {
    "Effectiveness": ["effective", "worked", "improved", "care", "treatment"],
    "Wait Time": ["wait", "waiting", "delay", "long"],
    "Staff": ["staff", "doctor", "nurse", "rude", "kind"],
    "Cleanliness": ["clean", "dirty", "hygiene"],
    "Cost": ["cost", "bill", "expensive", "insurance"],
    "Communication": ["explain", "communication", "informed"],
}

# ================= LOAD DATA =================
df = pd.read_excel(INPUT_FILE)
df[TEXT_COL] = df[TEXT_COL].astype(str).str.lower()
df[STATE_COL] = df[STATE_COL].astype(str)

# ================= HELPER FUNCTIONS =================
def contains_attribute(text, keywords):
    return any(k in text for k in keywords)

def sentiment_label(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

# ================= MAIN ANALYSIS =================
for cluster_name, states in CLUSTERS.items():

    cluster_df = df[df[STATE_COL].isin(states)]
    net_scores = {}

    for attribute, keywords in ATTRIBUTES.items():

        relevant = cluster_df[
            cluster_df[TEXT_COL].apply(lambda x: contains_attribute(x, keywords))
        ]

        if relevant.empty:
            net_scores[attribute] = 0.0
            continue

        sentiments = relevant[TEXT_COL].apply(sentiment_label)

        pos = (sentiments == "positive").sum()
        neg = (sentiments == "negative").sum()
        total = pos + neg

        net_scores[attribute] = 0.0 if total == 0 else ((pos - neg) / total) * 100

    # ================= PLOTTING =================
    attributes = list(net_scores.keys())
    values = list(net_scores.values())

    colors = [
        "green" if v > 0 else "red" if v < 0 else "gray"
        for v in values
    ]

    plt.figure(figsize=(5, 6))
    bars = plt.bar(attributes, values, color=colors)

    # Zero reference line
    plt.axhline(0, color="black", linewidth=1)

    # Dynamically space y-limits so labels never touch axes
    max_val = max(abs(v) for v in values) if values else 1
    padding = max(10, max_val * 0.25)
    plt.ylim(-max_val - padding, max_val + padding)

    # Value labels – always OUTSIDE bars, never on axis
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2

        if value > 0:
            y = value + padding * 0.15
            va = "bottom"
        elif value < 0:
            y = value - padding * 0.15
            va = "top"
        else:
            y = padding * 0.15
            va = "bottom"

        plt.text(
            x,
            y,
            f"{value:.1f}%",
            ha="center",
            va=va,
            fontsize=11
        )

    plt.ylabel("Net Score (%)", fontsize=13)
    plt.title(f"Attribute Net Scores — {cluster_name}", fontsize=13)
    plt.xticks(rotation=30, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    output_path = f"{OUTPUT_DIR}/{cluster_name}_attribute_net_score.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {output_path}")

print("🎉 Attribute Net Score analysis completed with clean label placement.")
