import pandas as pd
from utils import detect_emojis, filter_nouns_spacy, get_product_with_max_reviews, map_lda_to_general_topic, optimal_topic_number, split_into_sentences, tokenize_reviews_to_sequences, get_product_with_n_reviews, lemmatisation_stopwords_series
import nltk
import spacy

# Add the GuidedLDA directory to the Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'GuidedLDA')))

from guidedlda.guidedlda import GuidedLDA
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


####################### LDA #######################

def apply_guidedlda(sequences, seed_topic_list):
    """
    Apply GuidedLDA to extract topics from the reviews using seed topic list.

    Args:
        sequences (list): list/series of tokenized sequences 
        seed_topic_list (list): list of seed topics to guide the model.

    Returns:
        topics_df (pd.DataFrame): DataFrame with the topics for each document.
    """

    # vectorization
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(sequences)

    # vocab in the reviews
    vocab = vectorizer.get_feature_names_out()
    word2id = vectorizer.vocabulary_

    seed_topics = {}
    for t_id, st in enumerate(seed_topic_list):
        for word in st:
            if word in word2id:
                seed_topics[word2id[word]] = t_id
            else:
                print(f"Warning: The word '{word}' is not in the vocabulary and will be skipped.")

    model = GuidedLDA(
        n_topics=4,
        n_iter=100,
        random_state=7,
        refresh=10
    )

    # adjust guidedLDA with seed words
    model.fit(
        dtm,
        seed_topics=seed_topics,
        seed_confidence=0.15  
    )

    # get topic's distributions
    topic_word = model.topic_word_

    # get the top words for each topic
    print("Topics found by GuidedLDA (in training)")
    n_top_words = 15
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    # get the most probable topic for each review
    with_docs = model.transform(dtm)

    # create a dataframe with the topics for each document
    topics_df = pd.DataFrame(with_docs, columns=[f"Tópico {i}" for i in range(4)])  # Change 4 to 5

    # get the most probable topic for each document
    topics_df['topic'] = topics_df.idxmax(axis=1)

    return model