import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'D:\Code\Python\ML\dataasets\spam.csv')

X = df.iloc[:,1].values
y = df.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

vectorization = CountVectorizer()

X_train = vectorization.fit_transform(X_train)
X_test = vectorization.transform(X_test)

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

plt.figure(figsize = (6, 4))
df.iloc[:,:1].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.title('Pie chart')
plt.show()