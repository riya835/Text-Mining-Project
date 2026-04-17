# Fake Review Detection Using Naive Bayes

> Implementation of the paper **"Fake Review Detection Using Naive Bayes for Text Mining"**  
> Dataset: [Deceptive Opinion Spam Corpus](https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus) — Ott et al., ACL 2011

---

## Overview

This project builds a full text-classification pipeline to detect fake (deceptive) hotel reviews. Two Naive Bayes model variants are trained and evaluated on 800 labelled reviews (400 fake, 400 genuine), achieving **~88% accuracy** — substantially above the ~57% human baseline.

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Multinomial NB (Bag-of-Words) | ~86% | ~0.86 | ~0.93 |
| **Complement NB (TF-IDF)** | **~88%** | **~0.88** | **~0.94** |
| Human baseline (Ott et al.) | ~57% | — | — |

---

## Project Structure

```
.
├── fake_review_detection.ipynb   # Main Jupyter notebook (full pipeline)
├── README.md                     # This file
└── (outputs generated at runtime)
    ├── eda_plots.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    └── top_features_*.png
```

---

## Requirements

- Python 3.10+
- Jupyter Notebook or JupyterLab

Install all dependencies with:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

## Dataset Setup

The notebook uses the **Deceptive Opinion Spam Corpus**.

1. Go to https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus
2. Download and extract the ZIP
3. Place the CSV at: `deceptive-opinion-spam-corpus/deceptive_opinion_spam_corpus.csv`


---

## Quick Start

```bash
# 1. Clone / download this project
# 2. Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn jupyter

# 3. Launch Jupyter
jupyter notebook fake_review_detection.ipynb
# or
jupyter lab fake_review_detection.ipynb

# 4. Run all cells (Kernel → Restart & Run All)
```

---

## Pipeline Walkthrough

The notebook is structured into 12 sections:

| # | Section | Description |
|---|---|---|
| 0 | Install Dependencies | Auto-installs required packages |
| 1 | Imports & Configuration | Libraries, global settings, random seed |
| 2 | Dataset Loading | Loads real corpus or synthetic fallback |
| 3 | EDA | Class balance, word count distributions, box plots |
| 4 | Text Preprocessing | Lowercase → strip punctuation → collapse whitespace |
| 5 | Feature Extraction | BoW and TF-IDF with unigrams + bigrams (top 10,000 tokens) |
| 6 | Model Training | MultinomialNB (α=0.5) + ComplementNB (α=0.3) |
| 7 | Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices |
| 8 | Cross-Validation | 5-fold stratified CV with full pipelines |
| 9 | Feature Importance | Top discriminative tokens per class (bar charts) |
| 10 | Summary Table | Side-by-side model comparison with human baseline |
| 11 | Live Prediction | `predict_review()` function for new inputs |

---

## How It Works

### Naive Bayes Decision Rule (log-space)

```
log P(c|d) ∝ log P(c) + Σ log P(wᵢ | c)
```

With **Laplace (add-α) smoothing** to handle unseen words:

```
P(wᵢ | c) = (count(wᵢ, c) + α) / (Σ count(wⱼ, c) + α × |V|)
```

- **MultinomialNB** — models discrete token frequencies (BoW features), α=0.5  
- **ComplementNB** — trains on the complement class (better for TF-IDF), α=0.3

### Linguistic Signals

| FAKE reviews | GENUINE reviews |
|---|---|
| Superlatives & intensifiers (`absolutely`, `magical`) | Hedged language (`decent`, `okay`) |
| Generic praise (`highly recommended`) | Specific complaints (`noisy`, `broken`) |
| Emotional amplification (`breathtaking`) | Concrete facility details (Wi-Fi, parking) |

---

## Live Prediction Example

```python
from fake_review_detection import predict_review   # or run in the notebook

predict_review(
    "Absolutely perfect stay! Every single aspect was magical and truly breathtaking. "
    "Highly recommended to everyone!"
)
# → MNB: FAKE  (fake prob: 0.91)
# → CNB: FAKE  (fake prob: 0.93)
```

---

## Limitations

- **Domain-specific**: trained on hotel reviews only; may not generalise to other product categories
- **Word-order agnostic**: bag-of-words features discard context and sentence structure
- **Binary classification**: cannot handle *partially* deceptive content
- **Adversarial fragility**: sophisticated fake reviews mimicking genuine language patterns may evade detection

---

## Future Work

- Fine-tune **BERT / RoBERTa** for contextual representations
- Train on **mixed-domain corpora** (Amazon, Yelp) for better generalisation
- **Ensemble methods**: combine NB with SVM, Random Forest, XGBoost
- Add **reviewer behaviour features** (account age, posting frequency)
- Integrate **SHAP / LIME** for prediction explainability

---

## References

1. M. Ott, Y. Choi, C. Cardie, and J. T. Hancock, "Finding Deceptive Opinion Spam by Any Stretch of the Imagination," *ACL 2011*.
2. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, 2011.
3. J. D. M. Rennie et al., "Tackling the Poor Assumptions of Naive Bayes Text Classifiers," *ICML 2003*.
4. Kaggle Dataset: https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus

---

## License

For academic and educational use. Dataset is subject to its original licence (Ott et al., 2011).

