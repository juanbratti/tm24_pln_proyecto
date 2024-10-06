import pandas as pd
from scripts.preprocessing import load_and_preprocess_data, tokenize_reviews, vectorize_sequences
from scripts.topic_modelling import apply_bertopic, visualize

def main():

    # fp to parsed input file
    processed_file_path = '../data/parsed_input_file.csv'

    # preprocessing of reviews
    dataset, reviews_cleaned, product_id = load_and_preprocess_data(processed_file_path, 40)

    # tokenizetion of reviews
    sequences_list, sequences_series = tokenize_reviews(reviews_cleaned) 
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
    print(topic_model.get_document_info(sequences_list))
    print(topic_model.get_representative_docs())
    print(topic_model.topic_labels_)
    print(topic_model.hierarchical_topics(sequences_list))
    print(topic_model.topic_aspects_)
    
    print("-----------------------------------------")

    visualize(topic_model, sequences_list)

if __name__ == "__main__":
    main()