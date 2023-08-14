import numpy as np
import pandas as pd
from scipy import stats

data = pd.read_csv("rare elements.csv")
measurements = data['concentration'].values

sample_size = int(input("Enter the sample size: "))
confidence_level = float(input("Enter the confidence level (between 0 and 1): "))
desired_precision = float(input("Enter the desired level of precision: "))

sample_mean = np.mean(measurements[:sample_size])
sample_std = np.std(measurements[:sample_size], ddof=1)

t_score = stats.t.ppf(1 - (1 - confidence_level) / 2, df=sample_size - 1)

margin_of_error = t_score * sample_std / np.sqrt(sample_size)

confidence_interval_upper = sample_mean + margin_of_error
confidence_interval_lower = sample_mean - margin_of_error

required_sample_size = ((t_score * sample_std) / desired_precision)**2

print("Sample mean:", sample_mean)
print("Confidence interval:", (confidence_interval_lower, confidence_interval_upper))
print("Required sample size for desired precision:", int(np.ceil(required_sample_size)))
