# DA5401 — End-Semester Data Challenge
**Student Name:** Praveen B  
**Roll Number:** MM22B014  
**Course Code:** DA5401  
**Assignment:** End-Semester Data Challenge — Two-Tower Regression Model for Metric Score Prediction

---

## 📌 Summary of Work

- Worked on the DA5401 End-Semester Data Challenge to predict **metric scores (0–10)** for prompt–response pairs across **145 safety and quality metrics**.
- Combined text fields (`prompt`, `response`, `system_prompt`) into a single merged text input.
- Generated **768-dimensional text embeddings** using the multilingual transformer model  
  *paraphrase-multilingual-mpnet-base-v2*.
- Loaded and aligned the provided **metric name embeddings** for each training and test instance.
- Built a **Two-Tower Neural Network** architecture:
  - **Metric Tower** for processing metric embeddings  
  - **Text Tower** for processing merged text embeddings  
  - Fusion layer uses concatenation of transformed embeddings and pairwise features (`|M - T|` and `M * T`)
- Added **synthetic "hard negative" samples** by mismatching high-score text examples with random metrics to improve discriminative learning.
- Trained the model using **5-Fold Cross-Validation** and a **Focal Weighted MSE Loss** to focus more on large prediction errors.
- Averaged test predictions from all folds and clipped outputs to the range **[0, 10]** to generate final scores.

---

## 📂 Dataset Overview

- **train_data.json** — Contains labeled examples with prompts, responses, system prompts, metric names, and true scores.
- **test_data.json** — Same structure as training data but without the score column.
- **metric_names.json** — List of all 145 metric names.
- **metric_name_embeddings.npy** — Pre-computed 768-dimensional embeddings for each metric.

These datasets together define a supervised regression task:  
**(metric_name + conversation text) → score**

---

## ✅ Key Outcomes

- Implemented a strong two-tower regression approach using transformer embeddings.
- Enhanced training using synthetic negative sampling and feature engineering.
- Achieved stable performance across folds and produced leaderboard-ready predictions.
