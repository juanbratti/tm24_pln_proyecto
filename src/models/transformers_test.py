from transformers import pipeline
from textblob import TextBlob  # Import for sentiment analysis (optional)

#  model
model_name = "facebook/bart-base"
# summarization pipeline
summarizer = pipeline("summarization", model=model_name)


# ex reviews
reviews = [
    "I love this product! It works perfectly and is exactly what I needed. Highly recommend to anyone looking for something reliable.",
    "The item arrived late and was damaged. The quality is also not what I expected, very disappointed with this purchase.",
    "Great value for the price. It exceeded my expectations and I would definitely buy it again. The customer service was also excellent.",
    "The product did not match the description and the color was completely off. I'm returning it as it's not usable."
]

summaries = []
for review in reviews:
    summary_length = min(50, len(review))
    summary = summarizer(review, max_length=summary_length)[0]['summary_text']
    # Optional Sentiment Analysis
    sentiment = TextBlob(summary).sentiment
    if sentiment.polarity > 0:
        summary += " (Positive)"  # Add sentiment indicator (optional)
    elif sentiment.polarity < 0:
        summary += " (Negative)"  # Add sentiment indicator (optional)
    summaries.append(summary)

general_summary = " | ".join(summaries)

print("General Summary of All Reviews:\n")
print(general_summary)