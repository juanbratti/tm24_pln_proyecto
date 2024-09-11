import pandas as pd
from utils import detect_emojis, get_product_with_max_reviews, parse_reviews_txt_to_csv, tokenize_reviews_to_sequences

# preprocess raw data into a pandas dataframe
file_path = '../data/Arts.txt'

# get filepath to parsed file
processed_file_path = parse_reviews_txt_to_csv(file_path)

# turn the csv file to a panda's dataframe
dataset = pd.read_csv(processed_file_path)

# get the product with the most reviews
product_id, max_reviews = get_product_with_max_reviews(processed_file_path)

# get the review text for the product with the most reviews
reviews = dataset[dataset['productId'] == product_id]['reviewText']

# detect emojis
if (detect_emojis(reviews)):
    print('There are emojis in the reviews')
else:
    print('There are no emojis in the reviews')

# tokenization in sequences of 5 words
sequences = tokenize_reviews_to_sequences(reviews,5)

print(sequences)