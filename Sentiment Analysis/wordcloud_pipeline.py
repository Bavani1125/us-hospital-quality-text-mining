# ============================================================
# Cluster-wise Word Cloud Generator (Yelp Reviews)
# Fully aligned with clustering methodology
# ============================================================

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# ================= DOWNLOAD NLTK RESOURCES =================
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ================= CONFIG =================
INPUT_FILE = "input/yelp_reviews.xlsx"
OUTPUT_DIR = "output/wordclouds"

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

# ================= TEXT CLEANING =================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)

# ================= LOAD DATA =================
df = pd.read_excel(INPUT_FILE)

# Ensure expected columns exist
assert TEXT_COL in df.columns, f"Missing column: {TEXT_COL}"
assert STATE_COL in df.columns, f"Missing column: {STATE_COL}"

# Clean text once (important for consistency)
df["clean_text"] = df[TEXT_COL].apply(clean_text)

# ================= WORD CLOUD PER CLUSTER =================
for cluster_name, states in CLUSTERS.items():

    cluster_df = df[df[STATE_COL].isin(states)]

    combined_text = " ".join(cluster_df["clean_text"].tolist())

    if len(combined_text.strip()) == 0:
        print(f"⚠️ No text found for {cluster_name}, skipping.")
        continue

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=150,
        colormap="tab10"
    ).generate(combined_text)

    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(
        f"Word Cloud — {cluster_name}\nStates: {', '.join(states)}",
        fontsize=14
    )

    output_path = f"{OUTPUT_DIR}/{cluster_name}_wordcloud.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {output_path}")

print("🎉 All cluster-wise word clouds generated successfully.")
