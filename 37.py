import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Sample data (replace this with your actual data)
data = {
    'study_time': [2, 3, 4, 5, 6, 2.5, 3.5, 4.5, 5.5, 6.5],
    'exam_scores': [60, 65, 70, 75, 80, 62, 68, 72, 78, 82]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Calculate basic statistics
study_time_mean = df['study_time'].mean()
study_time_median = df['study_time'].median()
study_time_std = df['study_time'].std()

exam_scores_mean = df['exam_scores'].mean()
exam_scores_median = df['exam_scores'].median()
exam_scores_std = df['exam_scores'].std()

# Calculate the Pearson correlation coefficient and p-value
corr_coeff, p_value = pearsonr(df['study_time'], df['exam_scores'])

# Data Visualization
plt.figure(figsize=(12, 6))

# Scatter Plot
plt.subplot(1, 2, 1)
plt.scatter(df['study_time'], df['exam_scores'])
plt.xlabel('Study Time (hours)')
plt.ylabel('Exam Scores')
plt.title('Scatter Plot of Study Time vs. Exam Scores')

# Regression Plot
plt.subplot(1, 2, 2)
sns.regplot(x='study_time', y='exam_scores', data=df)
plt.xlabel('Study Time (hours)')
plt.ylabel('Exam Scores')
plt.title('Regression Plot of Study Time vs. Exam Scores')

plt.tight_layout()
plt.show()

# Print statistics and correlation results
print(f'Study Time - Mean: {study_time_mean}, Median: {study_time_median}, Std: {study_time_std}')
print(f'Exam Scores - Mean: {exam_scores_mean}, Median: {exam_scores_median}, Std: {exam_scores_std}')
print(f'Pearson Correlation Coefficient: {corr_coeff}')
print(f'P-value: {p_value}')

# Interpretation
if corr_coeff > 0:
    print('There is a positive correlation between study time and exam scores.')
elif corr_coeff < 0:
    print('There is a negative correlation between study time and exam scores.')
else:
    print('There is no significant correlation between study time and exam scores.')

# Additional analysis and conclusions can be added based on your specific dataset and goals.
