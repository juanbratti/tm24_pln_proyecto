import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def summarize_review(review):
  """Summarizes a short review using keyword extraction and sentiment analysis."""

  # tokenization of the review
  tokens = word_tokenize(review)

  # removal of stopwords
  stop_words = set(nltk.corpus.stopwords.words('english'))
  filtered_tokens = [word for word in tokens if word not in stop_words]

  # Extract keywords using TF-IDF (optional)
  # You can use libraries like scikit-learn for TF-IDF

  # sentiment analysis
  sid = SentimentIntensityAnalyzer()
  sentiment = sid.polarity_scores(review)

  # summary creation
  summary = " ".join(filtered_tokens[:5])  # Take the first 5 words
  if sentiment['compound'] > 0:
    summary += " (Positive)"
  elif sentiment['compound'] < 0:
    summary += " (Negative)"
  else:
    summary += " (Neutral)"

  return summary

# examples
reviews = [
  "The food was delicious!",
  "The service was terrible.",
  "The atmosphere was nice, but the food was okay."
]

for review in reviews:
  print(summarize_review(review))