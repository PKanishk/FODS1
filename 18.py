import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as py

age=np.array([23,23,27,27,39,41,47,49,50,52,54,54,56,57,58,58,60,61])
fat=np.array([9.5,26.5,7.8,17.8,31.4,25.9,27.4,27.2,31.2,34.6,42.5,28.8,33.4,30.2,34.1,32.9,41.2,35.7])

print(" the mean of age is:",age.mean())
print("the standard of age is:",age.std())

print(" the mean of fat is:",fat.mean())
print("the standard of fat is:",fat.std())


plt.bar(age,fat)
plt.xlabel('age')
plt.ylabel('fat')
plt.title(' box plot for age and fat')

plt.scatter(age,fat)
plt.xlabel('age')
plt.ylabel('fat')
plt.title(' scatter plot for age and fat')


plt.figure(figsize=(8, 6))
stats.probplot(age, dist="norm", plot=plt)
plt.title('QQ Plot for age')

plt.figure(figsize=(8, 6))
stats.probplot(fat, dist="norm", plot=plt)
plt.title('QQ Plot for fat')
plt.show()

