import pandas as pd
data=pd.read_csv("social media platform.csv")
df=data["likes"].value_counts()
print(df)