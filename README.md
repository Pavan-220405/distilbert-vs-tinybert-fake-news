### Open the PDF if .ipynb doesn't show up. You can download it for verification 

# Fake News Classification using DistilBERT and TinyBERT

This repository presents a comparative study of transformer-based models for fake news classification. The project fine-tunes and evaluates two lightweight BERT variants — DistilBERT and TinyBERT — to analyze their performance, efficiency, and suitability for real-world NLP classification tasks.

---

## Project Overview

Fake news detection is a critical NLP problem that requires both accuracy and efficiency. In this project, two compact transformer models are fine-tuned on a labeled fake news dataset and systematically compared based on classification performance and computational efficiency.

The project demonstrates an end-to-end NLP pipeline using Hugging Face Transformers, covering data preprocessing, model training, evaluation, and comparison.

---

## Dataset

- The dataset is sourced from **GitHub**
- Contains labeled news articles classified as real or fake
- Text-based binary classification problem
- Dataset is split into:
  - Training set
  - Validation set
  - Test set  
- Stratified splitting is used to preserve label distribution across splits

---

## Models Used

### DistilBERT
- A distilled version of BERT
- Retains most of BERT’s performance with fewer parameters
- Faster training and inference compared to BERT-base

### TinyBERT
- A heavily compressed BERT variant
- Optimized for low-latency and low-resource environments
- Smaller model size with reduced computational cost

---

## Workflow

1. **Environment Setup**
   - PyTorch
   - Hugging Face Transformers and Datasets
   - Scikit-learn
   - Pandas and NumPy

2. **Exploratory Data Analysis**
   - Dataset inspection
   - Class distribution analysis

3. **Data Splitting**
   - Stratified train, validation, and test split

4. **Dataset Conversion**
   - Conversion from Pandas DataFrame to Hugging Face `Dataset` and `DatasetDict`

5. **Label Encoding**
   - Creation of `label2id` and `id2label` mappings

6. **Tokenization**
   - Tokenization using respective model tokenizers
   - Padding and truncation applied
   - Removal of unnecessary columns for efficient training

7. **Model Fine-Tuning**
   - DistilBERT fine-tuned for sequence classification
   - TinyBERT fine-tuned for sequence classification
   - Training handled using Hugging Face `Trainer` API

8. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1-score

9. **Model Comparison**
   - Performance comparison between DistilBERT and TinyBERT
   - Analysis of accuracy vs model efficiency trade-offs

10. **Model Saving**
    - Fine-tuned models and tokenizers saved for inference or deployment

---

## Results Summary

- Both DistilBERT and TinyBERT demonstrate strong performance on the fake news classification task
- DistilBERT achieves higher classification accuracy and F1-score
- TinyBERT offers faster inference and lower memory usage with a small performance trade-off
- The comparison highlights the balance between model size, speed, and predictive performance

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Google Colab (GPU)

---
