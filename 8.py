import pandas as pd
data=pd.read_csv("price of sold product.csv")
c=data.sort_values(by="prices of each sold product",ascending=False)
d=c.head()
print("top 5 products that have been sold the most in the past month is",d)