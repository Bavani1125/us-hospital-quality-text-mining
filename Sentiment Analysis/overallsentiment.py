# ============================================================
# Overall Sentiment Analysis (Cluster-wise)
# Listerine-style sentiment volume visualization
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
OUTPUT_DIR = "output/overall_sentiment"

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

# ================= LOAD DATA =================
df = pd.read_excel(INPUT_FILE)
df[TEXT_COL] = df[TEXT_COL].astype(str).str.lower()
df[STATE_COL] = df[STATE_COL].astype(str)

# ================= SENTIMENT FUNCTION =================
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

    sentiments = cluster_df[TEXT_COL].apply(sentiment_label)

    positive_count = (sentiments == "positive").sum()
    negative_count = (sentiments == "negative").sum()

    # ================= PLOTTING =================
    labels = ["Positive", "Negative"]
    values = [positive_count, -negative_count]  # negative goes below axis
    colors = ["green", "red"]

    plt.figure(figsize=(6, 6))
    bars = plt.bar(labels, values, color=colors)

    plt.axhline(0, color="black", linewidth=1)

    # Value labels
    for bar, value in zip(bars, values):
        y = value + (50 if value > 0 else -50)
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{abs(value)}",
            ha="center",
            va="bottom" if value > 0 else "top",
            fontsize=10
        )

    plt.ylabel("Number of Reviews")
    plt.title(f"Overall Sentiment — {cluster_name}")
    plt.tight_layout()

    output_path = f"{OUTPUT_DIR}/{cluster_name}_overall_sentiment.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {output_path}")

print("🎉 Overall sentiment analysis completed successfully.")
