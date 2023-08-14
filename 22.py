import pandas as pd
import numpy as np
from scipy import stats

data = pd.read_csv('customer_reviews.csv')

category_data = data[data['product_category'] == 'soap']

sample_mean = category_data['rating'].mean()
sample_std = category_data['rating'].std()

confidence_level = 0.95

sample_size = len(category_data)
t_critical = stats.t.ppf((1 + confidence_level) / 2, df=sample_size - 1)

margin_of_error = t_critical * (sample_std / np.sqrt(sample_size))
confidence_interval =(sample_mean - margin_of_error, sample_mean + margin_of_error)

print("Sample Mean Rating:", sample_mean)
print("Confidence Interval:", confidence_interval)
