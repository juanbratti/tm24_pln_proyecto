import pandas as pd
from utils import clean_reviews, map_lda_to_general_topic, get_product_with_n_reviews, lemmatisation_stopwords_series, split_into_sentences, tokenize_reviews_to_sequences
from sklearn.feature_extraction.text import CountVectorizer


####################### DATA PREPROCESS #######################

def load_and_preprocess_data(file_path, product_review_count):
    """
    Loads raw file and preprocesses the data. It:
    - extracts the product with a specific number of reviews
    - cleans the reviews (removes emojis, contractions, special characters and extra whitespaces)
    - saves the processed data to a new file

    Args:
        file_path (str): path to the raw data file  
        product_review_count (int): number of reviews for the product to extract

    Returns:
        dataset (pd.DataFrame): dataset in the filepath as DataFrame
        reviews_cleaned (pd.Series): Cleaned reviews for the product with the specified number of reviews
        product_id (str): Product ID
    """

    # load of the dataset
    dataset = pd.read_csv(file_path)
    
    # retrieve the product with a specific number of reviews
    product_id = get_product_with_n_reviews(file_path, product_review_count)
    
    # extraction and cleaning the reviews
    reviews_raw = dataset[dataset['productId'] == product_id]['reviewText']
    reviews_cleaned = clean_reviews(reviews_raw)

    # save the file to a csv
    reviews_raw.to_csv(f'../data/processed/product_{product_id}_processed.csv', index=False)
    
    return dataset, reviews_cleaned, product_id

def tokenize_reviews(reviews_cleaned, tokens):
    """
    Tokenizes the cleaned reviews and returns a list of sequences.

    Args:
        reviews_cleaned (pd.Series): cleaned reviews (preprocessed reviews)
        
    Returns:
        sequences_list (list): python list of tokenized sequences
    """

    # lemmatization and stopword removal
    processed_series = lemmatisation_stopwords_series(reviews_cleaned)
    
    # tokenization of the clean reviews
    if tokens == 0:
        reviews_tokens = split_into_sentences(processed_series)
    else:
        reviews_tokens = tokenize_reviews_to_sequences(processed_series, tokens)

    # convertion of the tokenized series to a list of strings
    sequences_list = reviews_tokens.tolist()
    
    return sequences_list, reviews_tokens


def vectorize_sequences(sequences_list):
    """
    Vectorizes the sequences using CountVectorizer.

    Args:
        sequences_list (list): list of tokenized sequences

    Returns:
        vectorizer (CountVectorizer): trained CountVectorizer model
        dtm: document-term matrix
    """


    # vectorization
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(sequences_list)

    return vectorizer, dtm
