### Importing Necessary Libraries ###
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

### Load dataset ###
df = pd.read_csv("Breast Cancer Dataset.csv")

### Drop unnecessary columns ###
df_cleaned = df.drop(columns=["id", "Unnamed: 32"])

### Encode target variable (M=1, B=0) ###
label_encoder = LabelEncoder()
df_cleaned["diagnosis"] = label_encoder.fit_transform(df_cleaned["diagnosis"])

### Visualization 1: Class Distribution ###
plt.figure(figsize=(6, 4))
sns.countplot(x="diagnosis", data=df_cleaned)
plt.xticks([0, 1], ['Benign', 'Malignant'])
plt.title("Diagnosis Class Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.show()

### Visualization 2: Correlation Heatmap ###
plt.figure(figsize=(12, 10))
corr = df_cleaned.corr()
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

### Features and target ###
X = df_cleaned.drop(columns=["diagnosis"])
y = df_cleaned["diagnosis"]

### Train/Test split ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Standardize features ###
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### Train a logistic regression model ###
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)

### Predictions and probabilities ###
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

### Evaluation metrics ###
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

### Print evaluation ###
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
print("ROC-AUC Score:", roc_auc)

### Visualization 3: Confusion Matrix Heatmap ###
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

### Visualization 4: ROC Curve ###
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

### Visualization 5: Top 10 Features by Coefficient Magnitude ###
feature_names = X.columns
coef = model.coef_[0]
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef})
top_features = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x="Coefficient", y="Feature", data=top_features, palette="viridis")
plt.title("Top 10 Features by Logistic Regression Coefficients")
plt.show()
