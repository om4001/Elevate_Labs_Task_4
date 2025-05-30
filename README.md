---

````markdown
# 🔍 Breast Cancer Classification with Logistic Regression

This project demonstrates how to build a binary classification model using **Logistic Regression** on the **Breast Cancer Wisconsin Dataset**. It includes data preprocessing, model training, performance evaluation, and insightful visualizations.

---

## 📁 Dataset

- **Name**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source**: UCI Machine Learning Repository
- **Classes**: 
  - `M` = Malignant
  - `B` = Benign
- **Features**: 30 numeric features (mean, standard error, and worst of cell nuclei characteristics)

---

## 🎯 Objective

Build a binary classifier that can accurately predict whether a tumor is malignant or benign based on the input features using logistic regression.

---

## 🛠 Tools & Libraries

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📊 Project Workflow

1. **Data Loading & Cleaning**
   - Remove unnecessary columns (`id`, `Unnamed: 32`)
   - Encode labels (`M` → 1, `B` → 0)

2. **Data Visualization**
   - Class distribution plot
   - Correlation heatmap
   - Top features by model coefficients

3. **Preprocessing**
   - Train-test split
   - Feature standardization

4. **Model Building**
   - Logistic Regression using Scikit-learn

5. **Evaluation Metrics**
   - Confusion matrix & heatmap
   - Precision, recall, F1-score
   - ROC curve and AUC score

---

## 📈 Visualizations

- 📌 Diagnosis class distribution
- 📌 Correlation heatmap
- 📌 Confusion matrix heatmap
- 📌 ROC Curve
- 📌 Top 10 features by model coefficients

---

## ✅ Results Summary

- **Accuracy**: ~97%
- **ROC-AUC**: ~0.997
- **Very low false negatives and false positives**

---

## 💡 What You'll Learn

- How logistic regression works for binary classification
- How to evaluate classification models with metrics like precision, recall, F1-score, and ROC-AUC
- How to visualize model performance and data insights

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-logistic-regression.git
   cd breast-cancer-logistic-regression
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or Python script.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

---

```
