# Data Directory

This directory holds the raw and processed datasets used in this project.

### Dataset Source
- **Dataset**: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File**: `creditcard.csv` (~150MB, 284,807 transactions)

> [!NOTE]
> Due to GitHub's file size limit (100MB), `creditcard.csv` is excluded from git tracking via `.gitignore`.

### How to Download
You can download the dataset using the Kaggle API:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
```
Or directly download the ZIP from the [Kaggle Dataset Page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), extract it, and place `creditcard.csv` in this `data/` folder.
