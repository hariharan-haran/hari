import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon (if not already downloaded)
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Sample text for sentiment analysis
text = "I love this product! It's amazing."

# Analyze sentiment
sentiment_scores = sid.polarity_scores(text)

# Interpret the sentiment scores
compound_score = sentiment_scores['compound']

if compound_score >= 0.05:
    sentiment = "Positive"
elif compound_score <= -0.05:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

# Display the sentiment result
print(f"Sentiment: {sentiment}")
print(f"Compound Score: {compound_score}")
