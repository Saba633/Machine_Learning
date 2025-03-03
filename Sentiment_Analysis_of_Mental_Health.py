import pandas as pd
import praw
import time
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

# Reddit API credentials
reddit = praw.Reddit(
    client_id='LkAQXMjg-m3DCwARZAzS7A',
    user_agent='MentalHealthSentiment',
    client_secret='NCG0yzF8J45kDfbxlzUscW5rb3Ci1w'
)

subreddits = ['depression', 'anxiety', 'mentalhealth', 'stress']
num_posts = 500
post_data = []

# Scrape data from Reddit
for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.hot(limit=num_posts):
        post_data.append([
            subreddit_name,
            post.title,
            post.score,
            post.selftext,
            post.num_comments,
            post.created_utc
        ])
    print(f'Scraped {num_posts} posts from {subreddit_name}')
    time.sleep(2)

df = pd.DataFrame(post_data, columns=['subreddit', 'title', 'text', 'score', 'comments', 'timestamp'])
df.to_csv('reddit_mental_health.csv', index=False)
print("Data Saved")

# Text Cleaning Function
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = text.lower()
        words = word_tokenize(text)
        cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return " ".join(cleaned_words)
    else:
        return ""

# Sentiment Analysis Function
def get_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    compound_score = sentiment_score['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Load and preprocess data
data = pd.read_csv('reddit_mental_health.csv')
data.dropna(subset=['title', 'text'], inplace=True)
data.drop(columns=['timestamp'], inplace=True)
data["full_text"] = data["title"].astype(str) + " " + data["text"].astype(str)
data['clean_text'] = data['full_text'].apply(clean_text)
data['sentiment'] = data['clean_text'].apply(get_sentiment)
data.to_csv("sentimental_reddit_mental_health.csv", index=False)
print("Cleaned data saved successfully!")

# Sentiment Analysis Visualization
sentiment_counts = data["sentiment"].value_counts()
plt.figure(figsize=(6, 4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="coolwarm")
plt.xlabel("Sentiment Category")
plt.ylabel("Number of Posts")
plt.title("Sentiment Distribution")
plt.show()
