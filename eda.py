
import pandas as pd

# 1️⃣ Load dataset first
df = pd.read_csv(r"C:\Users\bia\Downloads\newsData\raw_analyst_ratings.csv")

# 2️⃣ Now you can check the columns
print("Columns in CSV:", df.columns)



import pandas as pd

df = pd.read_csv(r"C:\Users\bia\Downloads\newsData\raw_analyst_ratings.csv")
df.head()


# Headline length
df['headline_length'] = df['headline'].apply(len)
print(df['headline_length'].describe())

import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df['headline_length'], bins=30)
plt.title('Distribution of Headline Lengths')
plt.show()


publisher_counts = df['publisher'].value_counts()
print(publisher_counts.head(10))

publisher_counts.head(10).plot(kind='bar', title='Top 10 Publishers')
plt.ylabel('Number of Articles')
plt.show()

df['date'] = pd.to_datetime(df['date'], errors='coerce')

df.set_index('date', inplace=True)
articles_per_day = df.resample('D').size()

articles_per_day.plot(title='Articles per Day')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.show()



# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords


# Download stopwords for text analysis
nltk.download('stopwords')

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\bia\Downloads\newsData\raw_analyst_ratings.csv")

# -------------------------------
# Headline Length Analysis
# -------------------------------
df['headline_length'] = df['headline'].apply(len)
print("Headline Length Statistics:")
print(df['headline_length'].describe())

sns.histplot(df['headline_length'], bins=30)
plt.title('Distribution of Headline Lengths')
plt.show()

# -------------------------------
# Articles Per Publisher
# -------------------------------
publisher_counts = df['publisher'].value_counts()
print("\nTop 10 Publishers by Article Count:")
print(publisher_counts.head(10))

publisher_counts.head(10).plot(kind='bar', title='Top 10 Publishers')
plt.ylabel('Number of Articles')
plt.show()




from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
all_words = ' '.join(df['headline'].dropna()).lower().split()
filtered_words = [w for w in all_words if w not in stop_words]
word_counts = Counter(filtered_words)

print(word_counts.most_common(20))


# Step 1: Check the actual column names in your CSV
print(df.columns)



import matplotlib.pyplot as plt
import pandas as pd

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Set 'date' as the index
df.set_index('date', inplace=True)

# Count number of articles per day
articles_per_day = df.resample('D').size()

# Plot
articles_per_day.plot(title='Articles per Day')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.show()
