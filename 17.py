import pandas as pd
import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
data = pd.read_csv('customer feedback.csv')
all_feedback = ' '.join(data['feedback'])
translator = str.maketrans('', '', string.punctuation)
all_feedback = all_feedback.translate(translator)
tokens = word_tokenize(all_feedback)
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]
word_freq = Counter(filtered_tokens)
top_n = int(input("Enter the number of top words to display: "))
print(f"Top {top_n} most frequent words:")
for word, freq in word_freq.most_common(top_n):
    print(f"{word}: {freq}")
top_words = [word for word, _ in word_freq.most_common(top_n)]
top_freq = [freq for _, freq in word_freq.most_common(top_n)]
plt.bar(top_words, top_freq)
plt.xticks(rotation=45)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title(f'Top {top_n} Most Frequent Words')
plt.tight_layout()
plt.show()
