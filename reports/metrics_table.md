| Model Architecture | PR-AUC | ROC-AUC | Recall | Precision | F1-Score | False Positives (FP) | Missed Frauds (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Balanced)** | **0.8610** | 0.9580 | 76.53% | 96.15% | 0.8523 | 3 | 23 |
| **XGBoost (Weighted)** | **0.8553** | 0.9771 | 83.67% | 73.87% | 0.7847 | 29 | 16 |
| **XGBoost (Default)** | **0.8353** | 0.9634 | 78.57% | 91.67% | 0.8462 | 7 | 21 |
| **Logistic Regression (Baseline)** | **0.7388** | 0.9572 | 64.29% | 82.89% | 0.7241 | 13 | 35 |
| **Logistic Regression + SMOTE** | **0.7251** | 0.9712 | 91.84% | 5.89% | 0.1106 | 1,439 | 8 |
| **Logistic Regression (Class Weighted)** | **0.7175** | 0.9720 | 91.84% | 6.09% | 0.1142 | 1,388 | 8 |