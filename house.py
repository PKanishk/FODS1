import numpy as np
import pandas as pd
houses=pd.read_csv("house_data.csv")
a1=houses["sale price"]
prices=np.array(a1)
a2=houses["bedrooms"]
bedrooms=np.array(a2)
houses_with_more_than_4_bedrooms=bedrooms>4
filtered_sale_prices=prices[houses_with_more_than_4_bedrooms]
avg_sale_price=np.mean(filtered_sale_prices)
print(f"the average sale price of houses with more than four bedrooms is ${avg_sale_price:.2f}.")
