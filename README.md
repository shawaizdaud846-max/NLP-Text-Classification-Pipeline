# End-to-End NLP Pipeline: Document Classification and Word2Vec Prediction

## Overview
This repository contains a complete Natural Language Processing (NLP) pipeline developed as an academic lab assignment. The project processes a custom, self-collected corpus of 50 category-specific PDF documents, demonstrating the full NLP lifecycle: from raw text extraction and advanced text preprocessing to training word embeddings and machine learning classification models.
#Due to GitHub's file size limits, the raw PDF dataset is hosted here: https://drive.google.com/file/d/1mvtFMcT9ub9frVnHNH8U9qFPnDjqE5u4/view?usp=sharing

## Project Architecture
The pipeline is divided into three primary phases:

### 1. Data Extraction & Preprocessing
* **Extraction:** Automated raw text extraction from 50 categorical PDF files using `PyMuPDF`.
* **Cleaning:** Extensive text normalization using Regular Expressions and `NLTK`, including URL/email removal, punctuation stripping, tokenization, and stop-word filtering.
* **Corpus Generation:** Creation of distinct Unigram, Bigram, and Trigram feature sets, exported as a structured CSV (`cleaned_corpus.csv`).
* **Exploratory Data Analysis (EDA):** Top 20 N-gram frequency bar charts.

### 2. Part A: Next Word Prediction (Word Embeddings)
* **Modeling:** Trained Continuous Bag-of-Words (CBOW) and Skip-gram models using `Gensim`'s Word2Vec architecture.
* **Evaluation:** Models were trained and evaluated across unigram, bigram, and trigram configurations to predict contextually relevant words.
* **Visualizations:** A comparative training time analysis graph between CBOW and Skip-gram architectures.

### 3. Part B: Text Classification & Clustering
* **Supervised Learning (Classification):** Trained Logistic Regression and Multinomial Naive Bayes models using TF-IDF vectorization across all N-gram levels. Models are evaluated using Accuracy, Precision, Recall, and F1-Scores.
* **Unsupervised Learning (Clustering):** Applied K-Means clustering on the document vectors to observe natural document groupings and compare them against original folder labels.
* **Visualizations:** Model evaluation comparison graphs, a Confusion Matrix for the best-performing classifier, and a 2D PCA scatter plot visualizing K-Means clusters against actual categories.

## Dataset Structure
The raw dataset consists of 50 PDFs arranged in category-specific folders (acting as labels). The parsed, tokenized, and cleaned output used for model training is provided in the repository as `cleaned_corpus.csv`.

## Technology Stack
* **Language:** Python (Jupyter Notebook)
* **Text Processing:** `PyMuPDF` (fitz), `NLTK`, `Regex`
* **Machine Learning:** `scikit-learn` (Logistic Regression, Naive Bayes, K-Means, PCA, TF-IDF)
* **Word Embeddings:** `Gensim` (Word2Vec)
* **Data Visualization:** `Matplotlib`, `Seaborn`, `Pandas`
