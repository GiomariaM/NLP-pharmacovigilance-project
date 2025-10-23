# Adverse Drug Reaction Extraction with Transformers and CRFs

This project explores and compares modern Natural Language Processing (NLP) methods for detecting **Adverse Drug Reactions (ADRs)** in patient reviews.  
It replicates and extends the **ADRMine** model (Nikfarjam et al., 2015) by implementing both **Word2Vec embeddings with Conditional Random Fields (CRFs)** and **Transformer-based architectures (BERT)** for token classification.

---

## Overview

Pharmacovigilance aims to detect and understand adverse drug reactions that may not appear during clinical trials.  
This project proposes two main approaches to automate ADR extraction from unstructured patient reviews:

1. A **CRF-based model** using unsupervised word embeddings (Word2Vec or BERT feature-based).  
2. A **fine-tuned BERT model** for sequence labeling (token classification).

Both methods are benchmarked against standard information retrieval baselines such as **TF-IDF** and **BM25**.

---

## Datasets

The project uses several publicly available datasets:

* **CADEC** – annotated medical forum posts (Karimi et al., 2015)
* **PsyTAR** – psychiatric treatment reviews (Zolnoori et al., 2019)
* **ADR Dataset** – manually annotated patient drug reviews
* **UCI ML Drug Review Dataset** – large unannotated corpus for unsupervised embedding learning
* **Medical Term Lexicon** – 13,699 ADR terms used for lexicon-based retrieval

These datasets are **not included** in the repository due to licensing and size constraints.

---

## Methodology

### Baselines

* **TF-IDF** and **BM25** serve as information retrieval benchmarks.
  Each review is treated as a query, and ADR terms are ranked by relevance.

### ADRMine (Word2Vec + CRF)

* Constructs embeddings using **Word2Vec skip-gram** trained on unlabeled reviews.
* Performs **KMeans clustering** on the embeddings.
* Uses **cluster assignments and neighboring tokens** as CRF input features.

### ADRMine Extension (BERT Embeddings + CRF)

* Extracts **contextual embeddings** from a pre-trained BERT model.
* Averages the last four hidden layers to form token representations.
* Clusters embeddings and trains a CRF for binary sequence labeling.

### Fine-tuned BERT Model

* Uses **BERT for Token Classification** (`BertForTokenClassification` from Hugging Face).
* Labels each token as ADR, non-ADR, or padding.
* Fine-tuned on 70% of the annotated data, tested on the remaining 30%.
* Trained for 4 epochs using the Adam optimizer with learning rate `3e-5`.

---

## Usage

1. Prepare the datasets using `Creating_Dataset.ipynb`.
2. Generate embeddings using either:

   * `Word2Vec_Embeddings_and_CRF.ipynb`, or
   * `Bert_embeddings.ipynb`.
3. Train the models:

   * `CRF_Training.ipynb` for CRF-based models.
   * `BERT_Model_Final.ipynb` for fine-tuned BERT.
4. Use `Demo_final.ipynb` to test the trained models on new reviews.

---

## Report

The detailed methodology, results, and references are presented in `report.pdf`.
It includes data preprocessing details, architectural explanations, and model comparisons.

## Files

* `Creating_Dataset.ipynb` – Combines and standardizes annotated datasets
* `Preprocessing_fuction.ipynb` – Cleans and tokenizes raw text
* `Word2Vec_Embeddings_and_CRF.ipynb` – Word2Vec embeddings and CRF model
* `CRF_Training.ipynb` – Trains and evaluates CRF models
* `CRF_utils.py` – Helper functions for CRF
* `Bert_embeddings.ipynb` – Extracts BERT embeddings
* `BERT_Model_Final.ipynb` – Fine-tuning BERT for token classification
* `Baseline_results.ipynb` – TF-IDF and BM25 baselines
* `Demo_final.ipynb` – Model demonstration
* `report.pdf` – Full report and experimental results

