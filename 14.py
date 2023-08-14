import pandas as pd
data=pd.read_csv("price of sold product 14.csv")
df=data["age"].value_counts()
print(df)
