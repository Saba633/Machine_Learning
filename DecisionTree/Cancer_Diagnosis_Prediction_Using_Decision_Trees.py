import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Load dataset
data = pd.read_csv(r'D:\Code\Python\ML\dataasets\cancer_dataset.csv') # <-- Enter your dataset
df = pd.DataFrame(data)

# Split features and target
X = df.iloc[:, :-1]  # All columns except last one
y = df.iloc[:, -1]   # Last column as target

# Split into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

# Initialize Decision Tree Classifier
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Decision Tree Visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Decision Tree Visualization
plot_tree(model, feature_names=X.columns, class_names=[str(cls) for cls in model.classes_], 
          filled=True, rounded=True, fontsize=8, proportion=True, ax=axes[0])
axes[0].set_title("Decision Tree Visualization", fontsize=12)

# Feature Importance Plot
importances = model.feature_importances_
feature_names = X.columns

axes[1].barh(feature_names, importances, color='skyblue')
axes[1].set_xlabel("Feature Importance Score", fontsize=10)
axes[1].set_ylabel("Features", fontsize=10)
axes[1].set_title("Feature Importance in Cancer Prediction", fontsize=12)

# Adjust layout and display
plt.tight_layout()
plt.show()
