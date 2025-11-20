DA5401 — End-Semester Data Challenge

Student Name: Praveen B
Roll Number: MM22B014
Course Code: DA5401
Assignment: End-Semester Data Challenge — Two-Tower Regression Model for Metric Score Prediction

📌 Summary of Work

Worked on the 2025 DA5401 Data Challenge, where the task is to predict safety/quality metric scores (0–10 scale) for prompt–response pairs across 145 different evaluation metrics.

Preprocessed and combined dataset fields (prompt, response, system_prompt) into a single merged text input.

Used a multilingual transformer encoder (paraphrase-multilingual-mpnet-base-v2) to generate 768-dimensional text embeddings.

Loaded provided metric name embeddings and aligned them with each training instance.

Built a Two-Tower Neural Network architecture:

Metric tower processes metric embeddings.

Text tower processes text interaction embeddings.

Additional pairwise features (difference + elementwise product) enhance compatibility learning.

Introduced synthetic “hard negative” samples by mismatching high-score examples with random metrics to improve discriminative power.

Trained the model using 5-Fold Cross-Validation and a custom Focal Weighted MSE Loss to emphasize high-error cases.

Generated test predictions by averaging outputs from all folds and clipped scores to 
[
0
,
10
]
[0,10].

📂 Dataset Overview

train_data.json: Contains labeled examples with metric_name, prompts, responses, optional system prompt, and true score.

test_data.json: Same structure but without the score—requires prediction.

metric_names.json: Contains all 145 metric names used for evaluation.

metric_name_embeddings.npy: Pre-computed 768-dimensional embeddings corresponding to each metric.

These datasets collectively map:
(metric_name + conversation text) → score, forming the basis for supervised regression.

✅ Key Outcomes

Successfully implemented a robust metric–text matching model using transformer embeddings and neural regression.

Improved generalization through synthetic negatives and pairwise feature engineering.

Achieved stable out-of-fold RMSE and produced leaderboard-ready predictions
