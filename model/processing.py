import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

df = pd.read_csv("var/ASL/dataset_data.csv")

X = df.drop("label", axis=1)
y = df["label"]

normalize(X, "l1")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = [
    {
        "name": "RandomForestClassifier",
        "model": RandomForestClassifier(n_estimators=100, max_depth=5),
        "accuracy": [],
    },
    {
        "name": "LogisticRegression",
        "model": LogisticRegression(C=1.0, max_iter=1000),
        "accuracy": [],
    },
    {
        "name": "SVM",
        "model": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "accuracy": [],
    },
    {
        "name": "KNeighborsClassifier",
        "model": KNeighborsClassifier(n_neighbors=5),
        "accuracy": [],
    },
]

for model_info in models:
    model = model_info["model"]
    model_name = model_info["name"]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model: {model_name}, Accuracy: {accuracy}")
    model_info["accuracy"].append(accuracy)

print("\n")
print("_" * 30)
print("\n")
for model in models:
    print(f'Model: {model["name"]} Accuracy mean: {np.mean(model["accuracy"])}')
    with open("./model/accuracy.txt", "+a") as f:
        f.write(f'Model: {model["name"]} Accuracy mean: {np.mean(model["accuracy"])}\n')
        f.close()
