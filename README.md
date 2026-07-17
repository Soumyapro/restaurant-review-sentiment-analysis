# Restaurant Review Sentiment Analysis

A binary sentiment classifier that predicts whether a restaurant review is positive or negative, built with a classic NLP preprocessing pipeline and TF-IDF + linear models.

## Dataset

- **Source:** `Restaurant_Reviews.tsv` - 1,000 restaurant reviews
- **Columns:** `Review` (raw text), `Liked` (1 = positive, 0 = negative)
- 4 duplicate rows removed, leaving 996 reviews
- No missing values

## Pipeline

Text preprocessing was applied in the following order, chosen specifically to avoid common pitfalls in sentiment tasks (e.g. losing negation, misapplying lemmatization):

1. **Text cleaning** - lowercasing, HTML tag removal, contraction expansion (`"wasn't"` → `"was not"`), punctuation removal (while preserving `!` and `?`, which carry sentiment intensity)
2. **Tokenization** - `nltk.word_tokenize`
3. **Stopword removal** - NLTK's English stopword list, with negators (`not`, `no`, `nor`, `never`, `none`, `neither`) explicitly excluded so negation isn't stripped out
4. **Lemmatization** - POS-aware lemmatization using `WordNetLemmatizer`, tagging each word's part of speech first (via `pos_tag`) so verbs/adjectives/adverbs lemmatize correctly instead of defaulting to noun form

Cleaning happens on the raw sentence *before* tokenization, since contraction expansion needs full sentence context to work correctly.

## Feature Extraction

- **TF-IDF** (`TfidfVectorizer`) on the cleaned, lemmatized text
- 80/20 train/test split, stratified on the target to preserve class balance
- Vectorizer fit on the training set only, then applied to the test set (no data leakage)
- Resulting vocabulary: 1,491 features

## Models & Results

Two baseline classifiers were trained and compared:

| Model | Accuracy | Precision (0 / 1) | Recall (0 / 1) | F1 (0 / 1) |
|---|---|---|---|---|
| **Logistic Regression** | **83.5%** | 0.83 / 0.84 | 0.84 / 0.83 | 0.84 / 0.83 |
| Multinomial Naive Bayes | 78.5% | 0.81 / 0.77 | 0.75 / 0.82 | 0.78 / 0.79 |

**Logistic Regression is the better model** - it's more accurate overall and more balanced across classes. Naive Bayes shows a mild bias toward predicting "Liked," with lower recall on negative reviews (0.75 vs. 0.82).

## Tech Stack

- Python, pandas, numpy
- NLTK (tokenization, stopwords, WordNet lemmatizer, POS tagging)
- `contractions` (contraction expansion)
- scikit-learn (TF-IDF, Logistic Regression, Multinomial Naive Bayes, train/test split, evaluation metrics)
- matplotlib, seaborn (visualization)

## Project Structure

```
.
├── restaurant_review_sentiment_analysis.ipynb   # main notebook
├── Restaurant_Reviews.tsv                        # dataset
└── README.md
```
