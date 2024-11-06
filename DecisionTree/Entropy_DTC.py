import pandas as pd

data = pd.read_csv('/content/train.csv')

# print(data.columns.to_list)
# data.head()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier, plot_tree

model = DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

from matplotlib import pyplot as plt

plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=X.columns, filled=True, fontsize=10, precision=2)
plt.show()
