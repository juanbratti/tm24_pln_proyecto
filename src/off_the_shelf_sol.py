import pandas as pd
from utils import detect_emojis, get_product_with_max_reviews, tokenize_reviews_to_sequences, get_product_with_n_reviews, lemmatisation_stopwords_series
from transformers import pipeline

# get filepath to parsed file
processed_file_path = '../data/parsed_input_file.csv'

# turn the csv file to a panda's dataframe
dataset = pd.read_csv(processed_file_path)

# get the product with the most reviews
product_id = get_product_with_n_reviews(processed_file_path,10)

# get the review text for the product with the most reviews
reviews = (dataset[dataset['productId'] == product_id])['reviewText']

reviews_list = reviews.to_list()

summarizer = pipeline("summarization")

combined_reviews = " ".join(reviews_list)

print(combined_reviews)
    
# Define the prompt
prompt = (
    "Identify the product being mentioned in the opinions, and give me a brief summary of the product's quality "
    "according to the opinions. Mention a few of the product's characteristics and the opinions related to that "
    "characteristic."
)

input_text = f"{prompt}\n\nReviews:\n{combined_reviews}"

summary = summarizer(input_text, max_length=500, min_length=150, do_sample=False)[0]['summary_text']

print(summary)

print(combined_reviews)