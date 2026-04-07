# US Hospital Quality — Text Mining and NLP Analysis

MSc Thesis | Long Island University | 2026  
Author: Bavani Krishnamurthy Jothi

---

## Overview

This repository contains the analytical work supporting my MSc thesis on hospital quality and patient experience in the United States. The study draws on three independent public-domain sources and applies a combination of clustering and natural language processing techniques to examine patterns across a selected set of states.

The repository is organised to reflect the analytical pipeline as it was developed — from data preprocessing through to the final sentiment analysis outputs.

---

## Repository Structure

```
us-hospital-quality-text-mining-nlp/
├── data/processed/          # Aggregated and cleaned data used in analysis
├── scripts/
│   ├── correlation/         # Pearson correlation analysis
│   ├── scatter_matrix/      # Polynomial trendline visualisations
│   ├── clustering/          # Hierarchical and K-Means clustering
│   └── sentiment/           # VADER, NRCLex, passion index, word clouds
└── outputs/                 # Generated figures and charts
```

---

## Data Sources

All datasets used in this study are publicly available and fully de-identified prior to access. Raw source files are not included in this repository.

- CMS Hospital General Information (data.cms.gov)
- CMS OAS CAHPS Ambulatory Surgical Centre Survey (data.cms.gov)
- Yelp Open Dataset (business.yelp.com/data/resources/open-dataset)

---

## Methods

The analytical pipeline covers descriptive statistics, correlation analysis, hierarchical and K-Means clustering, and multi-dimensional sentiment analysis. Specific methodological details are documented in the thesis itself and are not reproduced here.

---

## Dependencies

All scripts are written in Python 3.11. Key libraries include:

- pandas, numpy, scipy, scikit-learn
- matplotlib, seaborn
- nltk (VADER), nrclex
- wordcloud

---

## License and Use

Copyright (c) 2026 Bavani Krishnamurthy Jothi. All rights reserved.

The code and outputs in this repository are shared for transparency and academic reference only. No part of this work — including scripts, figures, analysis, or written content — may be reproduced, adapted, or used in any form without explicit written permission from the author.

This repository does not constitute publication of the thesis findings.
