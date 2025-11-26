
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



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords


# Load dataset
df = pd.read_csv(r"C:\Users\bia\Downloads\newsData\raw_analyst_ratings.csv")

# Download stopwords
nltk.download('stopwords')

# Headline Length Analysis
# -------------------------------
df['headline_length'] = df['headline'].apply(len)
print("Headline Length Statistics:")
print(df['headline_length'].describe())

sns.histplot(df['headline_length'], bins=30)
plt.title('Distribution of Headline Lengths')
plt.xlabel('Headline Length')
plt.ylabel('Frequency')
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

# -------------------------------
# Keyword / Text Analysis (Top Words)
# -------------------------------
stop_words = set(stopwords.words('english'))
all_words = ' '.join(df['headline'].dropna()).lower().split()
filtered_words = [w for w in all_words if w not in stop_words]
word_counts = Counter(filtered_words)

print("\nTop 20 Most Common Words in Headlines:")
print(word_counts.most_common(20))

# -------------------------------
# Articles Over Time
# -------------------------------
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.set_index('date', inplace=True)
articles_per_day = df.resample('D').size()

articles_per_day.plot(title='Articles per Day')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv(r"C:\Users\bia\Desktop\week1\notebooks\data\raw_analyst_ratings.csv")


# Optional: convert 'date' column to datetime if not already
df['date'] = pd.to_datetime(df['date'], errors='coerce')


# Example: adding headline length as numeric feature
df['headline_length'] = df['headline'].apply(len)

# Compute correlation matrix
# Convert stock and headline_length to numeric, coercing errors to NaN
df['stock'] = pd.to_numeric(df['stock'], errors='coerce')
df['headline_length'] = pd.to_numeric(df['headline_length'], errors='coerce')

# Drop rows with NaN in these columns
df_clean = df.dropna(subset=['stock', 'headline_length'])

# Compute correlation on cleaned data
correlation_matrix = df_clean[['stock', 'headline_length']].corr()
print("Correlation Matrix:\n", correlation_matrix)

print("Correlation matrix:\n", correlation_matrix)

# Visualize correlation
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# Function to calculate sentiment polarity
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

# Apply to headline column
df['sentiment'] = df['headline'].apply(get_sentiment)

# Basic statistics
print(df['sentiment'].describe())

# Plot sentiment distribution
sns.histplot(df['sentiment'], bins=50, kde=True)
plt.title("Headline Sentiment Distribution")
plt.xlabel("Sentiment Polarity")
plt.ylabel("Frequency")
plt.show()


sns.scatterplot(data=df, x='sentiment', y='stock')
plt.title("Sentiment vs Stock Value")
plt.xlabel("Sentiment Polarity")
plt.ylabel("Stock")


# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # coerce invalid dates

# Drop rows with invalid dates
df = df.dropna(subset=['date'])

# -------------------------
# 1. Articles per day
# -------------------------
articles_per_day = df.groupby(df['date'].dt.date).size()

plt.figure(figsize=(12,6))
articles_per_day.plot()
plt.title('Number of Articles Published Per Day')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.tight_layout()
plt.show()

# -------------------------
# 2. Articles per month
# -------------------------
articles_per_month = df.groupby(df['date'].dt.to_period('M')).size()

plt.figure(figsize=(12,6))
articles_per_month.plot(kind='bar')
plt.title('Number of Articles Published Per Month')
plt.xlabel('Month')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------
# 3. Top 5 publishers over time
# -------------------------
top_publishers = df['publisher'].value_counts().head(5).index
df_top = df[df['publisher'].isin(top_publishers)]

# Group by date and publisher
articles_top_publishers = df_top.groupby([df_top['date'].dt.date, 'publisher']).size().unstack()

plt.figure(figsize=(12,6))
articles_top_publishers.plot()
plt.title('Top 5 Publishers: Articles Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Ensure 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Group articles by month
articles_per_month = df.groupby(df['date'].dt.to_period('M')).size()

# Plotting
plt.figure(figsize=(12,6))
articles_per_month.plot(kind='line', marker='o')
plt.title('Number of Articles Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
