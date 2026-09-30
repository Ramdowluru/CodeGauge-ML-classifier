# CodeGauge — ML-Based Programming Problem Difficulty Classifier

## 📌 Overview

**CodeGauge** is a machine learning project that predicts the difficulty level of programming problems using historical Codeforces submission data.

The system analyzes programming problem-solving behavior and classifies each problem into one of three difficulty categories:

* **Easy**
* **Medium**
* **Hard**

The machine learning workflow is implemented using **Snowflake's native ML Classification model**, with Snowflake used for data processing, feature engineering, model training, prediction, and evaluation.

---

## 🎯 Problem Statement

Programming platforms contain thousands of problems with different levels of difficulty. However, determining the difficulty of a problem can be challenging when only problem-solving activity is considered.

CodeGauge uses historical submission behavior to learn patterns associated with programming problem difficulty and predicts the difficulty category of a problem based on these patterns.

---

## 🏗️ Project Architecture

```text
Codeforces Submission Data
          │
          ▼
   Data Preparation
       (Colab)
          │
          ▼
      Snowflake
          │
          ├── Data Storage
          │
          ├── Feature Engineering
          │
          ├── Train/Test Split
          │
          ├── ML Classification
          │
          ├── Prediction
          │
          └── Evaluation
          │
          ▼
   Difficulty Prediction
   Easy / Medium / Hard
```

---

## 📊 Dataset

The project uses Codeforces submission data together with Codeforces problem metadata.

### Submission Dataset

Source:

**Hugging Face — `denkCF/UsersCodeforcesSubmissionsEnd2024`**

The dataset contains approximately **17.6 million submissions**.

Important columns include:

* `handle`
* `rating_at_submission`
* `problem_rating`
* `id_of_submission_task`
* `verdict`
* `time`

### Problem Metadata

Problem information was obtained through the **Codeforces Problemset API**.

The metadata includes:

* Contest ID
* Problem index
* Problem name
* Tags
* Problem rating

After joining the datasets and removing records without problem ratings, the final datas
