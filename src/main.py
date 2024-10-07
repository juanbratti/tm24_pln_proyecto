import pandas as pd
from scripts.preprocessing import load_and_preprocess_data, tokenize_reviews, vectorize_sequences
from scripts.topic_modelling import apply_bertopic, visualize, print_model_info

def main():

    # fp to parsed input file
    processed_file_path = '../data/parsed_input_file.csv'

    # preprocessing of reviews

    params = {
        'file_path': processed_file_path,
        'product_review_count': 40,
        # delete?
        'nan': True, 
        'emojis': True,
        'contractions': True,
        'special_chars': True,
        'whitespaces': True,
        'stopwords': True,
        'lemmatization': True,
        'lowercase': True,
        'emails_and_urls': True,
        'nouns': False,
        'adj': False,
        'numbers': True,
        'most_frequent': 0
    }

    dataset, reviews_cleaned, product_id = load_and_preprocess_data(params)

    tokens = 5
    # 0 to tokenize in sentences
    # n>0 to tokenize in n-grams

    # tokenizetion of reviews
    sequences_list, sequences_series = tokenize_reviews(reviews_cleaned, tokens, params['stopwords'], params['lemmatization']) 
    # sequences_list is a python list
    # sequences_series is a pandas series
    seed_topic_list = [
        ["price", "cheap", "expensive", "value", "affordable", "cost-effective", "budget-friendly", "inexpensive", "pricy", "overpriced", "costly"],
        ["quality", "good", "bad", "durable", "high-quality", "low-quality", "well-made", "fragile", "sturdy", "weak", "reliable", "unreliable"],
        ["use", "easy", "difficult", "works", "intuitive", "counterintuitive", "straightfoward", "complicated", "efficient", "inefficient", "unreliable"],
        ["design", "nice", "ugly", "aesthetic", "stylish", "unstylish", "attractive", "modern", "outdated", "elegant", "tasteful", "tasteless"]
    ]
    model = "all-MiniLM-L6-v2" # o "umap"
    reduced_topics = 10
    # application of BERTopic  modeling
    topic_model = apply_bertopic(sequences_list, seed_topic_list, model, reduced_topics)

    print_model_info(topic_model, sequences_list)

    visualize(topic_model, sequences_list)

if __name__ == "__main__":
    main()