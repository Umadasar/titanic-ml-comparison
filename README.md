# Titanic ML Model Comparison 

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-orange)
![Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange)

---

##  Objective
The goal of this project is to **compare multiple machine learning algorithms** on the Titanic dataset to predict passenger survival.  
Models compared: Decision Tree, Random Forest, SVM, kNN, Naive Bayes.  
Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.

---

##  Dataset
- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)  
- **Target:** `Survived` (0 = No, 1 = Yes)  

---

<details>
<summary>️ Methodology (click to expand)</summary>

1. **Data Preprocessing**  
   - Dropped irrelevant columns (`PassengerId`, `Name`, `Ticket`, `Cabin`)  
   - Filled missing values in `Age` and `Embarked`  
   - Encoded categorical variables (`Sex`, `Embarked`)  
   - Scaled features for SVM and kNN  

2. **Train-Test Split**  
   - 80% training, 20% testing  
   - Stratified to maintain class balance  

3. **Model Training**  
   - Decision Tree, Random Forest, SVM, kNN, Naive Bayes  
   - Default hyperparameters  

4. **Evaluation Metrics**  
   - Accuracy, Precision, Recall, F1-score, ROC-AUC

</details>

---

##  Results

| Model          | Accuracy | Precision | Recall  | F1-score | ROC-AUC |
|----------------|----------|-----------|---------|----------|---------|
| Decision Tree  | 0.8324   | 0.8000    | 0.7536  | 0.7761   | 0.8168  |
| Random Forest  | 0.8156   | 0.7903    | 0.7101  | 0.7481   | 0.8454  |
| SVM            | 0.8156   | 0.8462    | 0.6377  | 0.7273   | 0.8387  |
| kNN            | 0.8156   | 0.7903    | 0.7101  | 0.7481   | 0.8441  |
| Naive Bayes    | 0.7821   | 0.7273    | 0.6957  | 0.7111   | 0.8328  |

---

##  Visualizations
- Bar chart: Compare Accuracy, Precision, Recall, F1-score across models  
- ROC Curves: Compare true positive rate vs. false positive rate for each model  

---

<details>
<summary> Conclusion (click to expand)</summary>

- **Decision Tree** achieved the highest accuracy (83.2%)  
- **Random Forest** achieved the best ROC-AUC (0.845), making it the most balanced model  
- **SVM** achieved the best precision (0.846), though recall was lower  
- **Naive Bayes** underperformed, likely due to mixed categorical/numerical data  

**Next Steps:**  
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)  
- Try ensemble methods like Gradient Boosting or XGBoost  
- Apply cross-validation for more robust evaluation  

</details>

---

## 📌 References
- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)  

