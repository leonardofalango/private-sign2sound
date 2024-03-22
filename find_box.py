import pandas as pd

df = pd.read_csv("var/ASL/complete_data.csv")
df_conc = pd.read_csv("var/ASL/data.csv")

df = pd.concat([df, df_conc], ignore_index=True)
df.to_csv("var/ASL/complete_data.csv", index=False)
print("Data saved")
