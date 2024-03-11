import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("var/ASL/dataset_data.csv")
print(df.head(5))
print(df['label'].unique()) # all letters

print(df['landmark_1'][0])
# rfc = RandomForestClassifier(random_state=0)
