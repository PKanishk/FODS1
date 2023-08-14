import pandas as pd
data=pd.read_csv("property data.csv")
c=data.groupby("location")
avg=c["price"].mean()
print("The average listing price of properties in each location",avg)
m4bhk=data["nb"]>4
count=len(data[m4bhk])
print("The number of properties with more than four bedrooms",count)
larea=data["area"].max()
print("The property with the largest area",larea)