import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=5)

df = pd.read_csv("var/ASL/complete_data.csv")

X = df.drop("label", axis=1)
y = df["label"]

clf.fit(X, y)

with open("./model.pkl", "wb") as f:
    pickle.dump(clf, f)
