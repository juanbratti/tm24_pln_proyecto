import pandas as pd
from utils import detect_emojis, filter_nouns_spacy, get_product_with_max_reviews, map_lda_to_general_topic, optimal_topic_number, split_into_sentences, tokenize_reviews_to_sequences, get_product_with_n_reviews, lemmatisation_stopwords_series
import nltk
import spacy

# Add the GuidedLDA directory to the Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'GuidedLDA')))

import guidedlda
import numpy as np

# get filepath to parsed file
processed_file_path = '../data/parsed_input_file.csv'

# turn the csv file to a panda's dataframe
dataset = pd.read_csv(processed_file_path)

# get the product with 20 reviews
product_id = get_product_with_n_reviews(processed_file_path,40)

# get the review text for the product 20 reviews
reviews = (dataset[dataset['productId'] == product_id])['reviewText']

# detect emojis
if (detect_emojis(reviews)):
    print('There are emojis in the reviews')
else:
    print('There are no emojis in the reviews')

#######################

# def filter_adjectives_spacy(text_series):
#     filtered_text = []
    
#     for review in text_series:
#         # Procesar el texto con spaCy para obtener las categorías gramaticales
#         doc = nlp(review)
        
#         # Filtrar palabras que son adjetivos (pos_ == 'ADJ' para adjetivos)
#         adjectives = [token.text for token in doc if token.pos_ == 'ADJ']
        
#         # Unir los adjetivos en una cadena de texto
#         filtered_text.append(" ".join(adjectives))
    
#     return pd.Series(filtered_text)


# reviews_without_nouns = filter_adjectives_spacy(reviews)


reviews_sentences = split_into_sentences(reviews)

processed_series = lemmatisation_stopwords_series(reviews_sentences)

sequences_list = tokenize_reviews_to_sequences(processed_series, 5)

####################### LDA #######################

# vectorization
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
dtm = vectorizer.fit_transform(sequences_list)

print("VOCABULARY IN THE REVIEWS")
vocab = vectorizer.get_feature_names_out()

word2id = vectorizer.vocabulary_

print(vocab)

from sklearn.decomposition import LatentDirichletAllocation

# Define general themes and their keywords
seed_topic_list = [["cheap", "expensive", "price", "value", "affordable", "cost-effective", "budget-friendly", "inexpensive", "pricy", "overpriced", "costly"],
    ["good", "bad", "quality", "durable", "high-quality", "low-quality", "well-made", "fragile", "sturdy", "weak", "reliable", "unreliable"],
    ["easy", "difficult", "use", "works", "intuitive", "counterintuitive", "straightfoward", "complicated", "efficient", "inefficient", "unreliable"],
    ["nice", "ugly", "design", "aesthetic", "stylish", "unstylish", "atractive", "modern", "outdated", "elegant", "tastefull", "tasteless"],
]

seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        if word in word2id:
            seed_topics[word2id[word]] = t_id
        else:
            print(f"Warning: The word '{word}' is not in the vocabulary and will be skipped.")

model = guidedlda.GuidedLDA(
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
n_top_words = 10
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

# get the most probable topic for each review
with_docs = model.transform(dtm)

# create a dataframe with the topics for each document
topics_df = pd.DataFrame(with_docs, columns=[f"Tópico {i}" for i in range(4)])  # Change 4 to 5

# get the most probable topic for each document
topics_df['topic'] = topics_df.idxmax(axis=1)


# print sequence and the most probable topic
for i in range(len(sequences_list)):
    print(f"Sequence: {reviews_sentences[i]}")
    print(f"Most probable topic: {topics_df['topic'][i]}")
    print()
