import pandas as pd

data = pd.read_csv("Titanic-Dataset.csv")

print(data.head())
print(data.info())

print(data.describe())

print(data.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Missing Age ko average age se fill karna
data["Age"] = data["Age"].fillna(data["Age"].mean())

# Gender ko number me convert karna
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

# Features aur Target
X = data[["Pclass", "Sex", "Age", "Fare"]]
y = data["Survived"]

# Train aur Test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model banana
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

# Survival Count Graph
data["Survived"].value_counts().plot(kind="bar")

plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.savefig("survival_count.png")
plt.show()

# Male vs Female Count Graph
data["Sex"].value_counts().plot(kind="bar")

plt.title("Male vs Female Passengers")
plt.xlabel("Gender (0 = Male, 1 = Female)")
plt.ylabel("Count")

plt.show()