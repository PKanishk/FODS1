import pandas as pd
import numpy as np

# Assuming you have a CSV file with temperature data in the following format:
# City, Date, Temperature
# City1, 2023-01-01, 10.5
# City1, 2023-01-02, 12.0
# ...

# Load the dataset into a Pandas DataFrame
data = pd.read_csv('temperature_data.csv')

# 1. Calculate the mean temperature for each city
mean_temperatures = data.groupby('City')['Temperature'].mean()

# 2. Calculate the standard deviation of temperature for each city
std_dev_temperatures = data.groupby('City')['Temperature'].std()

# 3. Determine the city with the highest temperature range
temperature_range = data.groupby('City')['Temperature'].max() - data.groupby('City')['Temperature'].min()
city_with_highest_range = temperature_range.idxmax()

# 4. Find the city with the most consistent temperature (lowest standard deviation)
city_with_lowest_std_dev = std_dev_temperatures.idxmin()

# Print the results
print("Mean Temperatures:")
print(mean_temperatures)

print("\nStandard Deviations:")
print(std_dev_temperatures)

print("\nCity with Highest Temperature Range:", city_with_highest_range)
print("City with Most Consistent Temperature:", city_with_lowest_std_dev)
