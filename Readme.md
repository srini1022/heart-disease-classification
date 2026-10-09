# Heart Disease Classification with Decision Trees and Random Forests

## Overview

This project applies machine learning classification algorithms to predict the presence of heart disease using clinical attributes. It implements and compares a Decision Tree Classifier and a Random Forest Classifier, analyzes overfitting, visualizes the decision tree, examines feature importance, and uses cross-validation and hyperparameter tuning to evaluate model performance.

## Objectives

- Explore and prepare the Heart Disease dataset.
- Build and visualize a Decision Tree Classifier.
- Analyze overfitting by controlling tree depth.
- Train and evaluate a Random Forest Classifier.
- Compare model performance using classification metrics.
- Identify important features using feature importance.
- Apply stratified cross-validation.
- Optimize hyperparameters using GridSearchCV.

## Dataset Description

The dataset is stored in `data/heart.csv`.

| Property | Description |
|---|---|
| Original records | 1,025 |
| Total columns | 14 |
| Input features | 13 |
| Target column | `target` |
| Missing values | None detected |

The dataset contains clinical attributes such as age, sex, chest pain type, resting blood pressure, cholesterol, maximum heart rate, and other health-related measurements.

### Target Variable

- `0` — Absence of heart disease
- `1` — Presence of heart disease

### Data Preparation

The original dataset contained exact duplicate rows. These were removed before splitting the data, leaving 302 unique records. The input features and target were separated, followed by a stratified 80:20 train-test split.

Duplicate removal substantially reduced the dataset size, so model performance should be interpreted cautiously.

## Methodology

### 1. Exploratory Data Analysis
- Inspected the dataset structure and sample records.
- Checked missing values and duplicate rows.
- Examined the target distribution.
- Analyzed feature correlations with the target.

### 2. Decision Tree Classification
- Trained a baseline Decision Tree Classifier.
- Evaluated training and testing accuracy.
- Examined the confusion matrix and classification report.
- Visualized the decision tree.

### 3. Overfitting Analysis
Compared different values of `max_depth` to understand the relationship between model complexity and generalization performance.

### 4. Random Forest Classification
Trained a Random Forest Classifier and compared its performance with the Decision Tree.

### 5. Feature Importance
Analyzed Random Forest feature importances to identify the features that contributed most to the model's predictions.

### 6. Cross-Validation
Used stratified 5-fold cross-validation to evaluate model performance across multiple training folds.

### 7. Hyperparameter Tuning
Used GridSearchCV to select hyperparameters for both classifiers based on cross-validation accuracy.

## Results

### Hyperparameter-Tuned Models

| Model | Best CV Accuracy | Test Accuracy |
|---|---:|---:|
| Decision Tree | 80.94% | 70.49% |
| Random Forest | 85.88% | 77.05% |

### Selected Hyperparameters

**Decision Tree**
- Criterion: `entropy`
- Maximum depth: `5`
- Minimum samples per leaf: `1`
- Minimum samples to split: `10`

**Random Forest**
- Number of trees: `100`
- Maximum depth: `5`
- Maximum features: `sqrt`
- Minimum samples per leaf: `1`
- Minimum samples to split: `2`

### Key Findings

- The unrestricted Decision Tree achieved 100% training accuracy, indicating potential overfitting.
- Limiting tree depth reduced model complexity.
- The tuned Random Forest achieved higher cross-validation and test accuracy than the tuned Decision Tree in this experiment.
- Cross-validation and test accuracy differed, demonstrating why model evaluation should not rely on a single split.
- Feature importance indicates how the model uses features for its predictions; it does not establish causation or clinical significance.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Structure

```text
heart-disease-classification/
├── data/
│   └── heart.csv
├── heart_disease_classification.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd heart-disease-classification
```

Replace `<YOUR_REPOSITORY_URL>` with your GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `heart_disease_classification.ipynb` and execute the cells in order. Keep `data/heart.csv` in its specified location so the notebook can load the dataset correctly.

## Limitations

- Removing duplicate records reduced the dataset from 1,025 to 302 rows.
- The available data does not establish whether repeated records represent the same individuals or legitimate repeated observations.
- Results may vary with different train-test splits and datasets.
- Feature importance does not imply causation.
- The dataset and models have not been established as suitable for clinical diagnosis.

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic tool and must not be used as a substitute for professional medical advice.

## Future Improvements

- Validate performance on a larger, independent dataset.
- Evaluate additional metrics such as precision, recall, F1-score, and ROC-AUC.
- Explore feature selection and model calibration.
- Compare additional classification algorithms.
- Investigate the provenance and meaning of duplicate records before deciding how they should be handled.
