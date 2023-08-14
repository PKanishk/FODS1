import pandas as pd
data=pd.read_csv("customer reviews.csv")
cr=data["customer reviews"].value_counts()
print(cr)
