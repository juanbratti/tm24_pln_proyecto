import pandas as pd
from scripts.preprocessing import load_and_preprocess_data, tokenize_reviews, vectorize_sequences
from scripts.topic_modelling import apply_bertopic

def main():

    # fp to parsed input file
    processed_file_path = '../data/parsed_input_file.csv'

    # preprocessing of reviews
    dataset, reviews_cleaned, product_id = load_and_preprocess_data(processed_file_path, 40)

    tokens = 0
    # 0 to tokenize in sentences
    # n>0 to tokenize in n-grams

    # tokenizetion of reviews
    sequences_list, sequences_series = tokenize_reviews(reviews_cleaned, tokens) 
    # sequences_list is a python list
    # sequences_series is a pandas series

    seed_topic_list = [
        ["price", "cheap", "expensive", "value", "affordable", "cost-effective", "budget-friendly", "inexpensive", "pricy", "overpriced", "costly"],
        ["quality", "good", "bad", "durable", "high-quality", "low-quality", "well-made", "fragile", "sturdy", "weak", "reliable", "unreliable"],
        ["use", "easy", "difficult", "works", "intuitive", "counterintuitive", "straightfoward", "complicated", "efficient", "inefficient", "unreliable"],
        ["design", "nice", "ugly", "aesthetic", "stylish", "unstylish", "attractive", "modern", "outdated", "elegant", "tasteful", "tasteless"]
    ]

    # application of BERTopic  modeling
    topic_model = apply_bertopic(sequences_list, seed_topic_list)

     # Display the topics found
    print("Topics discovered:")
    print(topic_model.get_topic_info())

if __name__ == "__main__":
    main()