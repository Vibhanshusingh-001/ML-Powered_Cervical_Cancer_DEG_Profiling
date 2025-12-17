
# Model Comparison Based on Confusion Matrix Metrics


## 1. Evaluation Matrices

### Support Vector Machine (SVM – RBF)
<<img width="555" height="347" alt="Evaluation_SVM" src="https://github.com/user-attachments/assets/daaf8e25-25d9-4212-b164-a7ba88a6071c" />




### XGBoost
<img width="555" height="347" alt="Evaluation_xgb" src="https://github.com/user-attachments/assets/104f02b2-b73b-43f0-adff-fb4c550d334c" />

### Random Forest
<img width="557" height="347" alt="Evaluation_rf" src="https://github.com/user-attachments/assets/cb79c7ea-ff7b-4209-bc5d-83f3489c7558" />


### Logistic Regression
<img width="555" height="350" alt="Evaluation_LR" src="https://github.com/user-attachments/assets/f8c5026e-a52c-4315-8d06-1843eefdee40" />

### k-Nearest Neighbors (kNN)
<img width="555" height="350" alt="knn_evaluation" src="https://github.com/user-attachments/assets/20912b11-c70f-494f-abd2-57438efb0db4" />


### Decision Tree
<img width="541" height="348" alt="Evaluation_dt" src="https://github.com/user-attachments/assets/3a11bcaf-d09d-4710-9f18-aff14de3f1d0" />

---

## 2. Complete Performance Metrics Comparison

| Metric | SVM | XGBoost | Random Forest | Logistic Reg. | kNN | Decision Tree |
|------|-----|---------|---------------|---------------|-----|---------------|
| Accuracy | 0.625 | 0.750 | 0.750 | 0.625 | 0.875 | 0.750 |
| 95% CI (Accuracy) | (0.2449, 0.9148) | (0.3491, 0.9681) | (0.3491, 0.9681) | (0.2449, 0.9148) | (0.4735, 0.9968) | (0.3491, 0.9681) |
| No Information Rate | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| P-value (Acc > NIR) | 0.3633 | 0.1445 | 0.1445 | 0.3633 | 0.03516 | 0.1445 |
| Cohen’s Kappa | 0.250 | 0.500 | 0.500 | 0.250 | 0.750 | 0.500 |
| McNemar’s Test p-value | 0.2482 | 0.4795 | 0.4795 | 1.0000 | 1.0000 | 0.4795 |
| Sensitivity (Recall) | 0.250 | 0.500 | 0.500 | 0.500 | 0.750 | 0.500 |
| Specificity | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 |
| Positive Predictive Value | 1.000 | 1.000 | 1.000 | 0.667 | 1.000 | 1.000 |
| Negative Predictive Value | 0.571 | 0.667 | 0.667 | 0.600 | 0.800 | 0.667 |
| Prevalence | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| Detection Rate | 0.125 | 0.250 | 0.250 | 0.250 | 0.375 | 0.250 |
| Detection Prevalence | 0.125 | 0.250 | 0.250 | 0.375 | 0.375 | 0.250 |
| Balanced Accuracy | 0.625 | 0.750 | 0.750 | 0.625 | 0.875 | 0.750 |

---
 # Collective Interpretation of Model Performance Values



## 3. Dataset-Level Context

- Total samples: 8  
- Class distribution:  
  - Cancer = 4  
  - Normal = 4  
- No Information Rate (NIR): 0.50  

Because the dataset is **balanced but extremely small**, metric fluctuations are expected. Therefore:
- Accuracy alone is unreliable
- Sensitivity, specificity, balanced accuracy, and agreement metrics are important.



## 4. Accuracy and Balanced Accuracy 

- Accuracy ranges from **0.625 to 0.875**
- Balanced Accuracy follows the same pattern

 Meaning:
- All models perform **better than random guessing**
- Only one model (kNN) shows a **clear and meaningful improvement**
- Models with identical accuracy can still behave very differently in terms of clinical risk



## 5. Sensitivity vs Specificity 

### Specificity
- Most models achieve **perfect specificity (1.00)**
- This means **no normal samples are falsely diagnosed as cancer**

Collective implication:
- Models are conservative and avoid false alarms
- This inflates accuracy and precision

### Sensitivity
- Sensitivity ranges from **0.25 to 0.75**
- Several models miss **50–75% of cancer cases**

Collective implication:
- High specificity combined with low sensitivity indicates **systematic under-detection of cancer**
- This is dangerous in medical screening contexts

Key interpretation:
- Models prefer predicting “Normal” unless highly confident



## 6. Precision (PPV) and Negative Predictive Value (NPV)

### Precision (PPV)
- Mostly **1.00** across models

Interpretation:
- When a model predicts “Cancer”, it is almost always correct

### Negative Predictive Value (NPV)
- Ranges from **0.57 to 0.80**

Interpretation:
- A “Normal” prediction is less reliable
- False negatives are common, lowering trust in negative results

Collective insight:
- Models are good at confirming cancer, poor at ruling it out



## 7. Cohen’s Kappa (Agreement Beyond Chance)

- Kappa values range from **0.25 to 0.75**

Interpretation:
- Low Kappa (0.25): Weak learning beyond chance
- Moderate Kappa (0.50): Partial learning
- High Kappa (0.75): Strong signal extraction

Collective insight:
- Only one model demonstrates **robust pattern learning**
- Others rely heavily on class imbalance and conservative prediction



## 8. Statistical Significance (NIR and McNemar’s Test)

### Accuracy vs NIR
- Only one model significantly outperforms random guessing
- Others fail to achieve statistical confidence

### McNemar’s Test
- All models show **non-significant p-values**

Interpretation:
- Differences between models are not statistically decisive
- Expected given small sample size

Collective conclusion:
- Results indicate **trends**, not definitive superiority



## 9. Detection Rate and Detection Prevalence

- Detection rates are low across most models
- Detection prevalence often underestimates true prevalence (0.50)

Interpretation:
- Models under-predict cancer
- Conservative bias dominates prediction behavior



## 10. Collective Model Behavior Pattern

Across all metrics, models show a consistent pattern:

- Strong bias toward predicting “Normal”
- High confidence only when predicting “Cancer”
- Limited generalization due to data scarcity
- Apparent performance driven more by **error avoidance** than **true learning**



## 11. Overall Collective Interpretation

When interpreted together, these values indicate that:

- The modeling framework is **risk-averse**
- Most models avoid false positives at the cost of false negatives
- Only one model balances sensitivity and specificity effectively
- Agreement metrics confirm limited learning in most models
- Statistical tests caution against over-interpretation



## 12. Practical Implication

- Suitable for **method comparison or proof-of-concept**
- Not suitable for clinical decision-making
- Larger datasets and resampling strategies are essential
- Sensitivity-oriented optimization should be prioritized



## 13. Final Collective Conclusion

Taken as a whole, these metrics reveal that the models are **accurate but not clinically safe**, except one.  
The dominant issue is **missed cancer cases**, not false alarms.  
Thus, **balanced accuracy, sensitivity, and kappa** are more informative than raw accuracy in this study.


