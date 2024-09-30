import pandas as pd
from utils import detect_emojis, filter_nouns_spacy, get_product_with_max_reviews, map_lda_to_general_topic, optimal_topic_number, split_into_sentences, tokenize_reviews_to_sequences, get_product_with_n_reviews, lemmatisation_stopwords_series
import nltk
import spacy

# get filepath to parsed file
processed_file_path = '../data/parsed_input_file.csv'

# turn the csv file to a panda's dataframe
dataset = pd.read_csv(processed_file_path)

# get the product with 20 reviews
product_id = get_product_with_n_reviews(processed_file_path,20)

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

# transform series into list of strings
sequences_list = processed_series.tolist()

####################### LDA #######################

# vectorization
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
dtm = vectorizer.fit_transform(sequences_list)

from sklearn.decomposition import LatentDirichletAllocation

# Apply LDA
num_topics=4
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=0)
lda_model.fit(dtm)

# Title of the product
print("Product Title")
print(dataset[dataset['productId'] == product_id]['productTitle'].iloc[0])

# Print the topics found by the LDA model
print("Topics found via LDA (in training):")
for i, topic in enumerate(lda_model.components_):
    print(f"Topic {i}:")
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print("\n")

# get the topic probability distribution for each review
with_docs = lda_model.transform(dtm)

# dataframe with the topic probability distribution for each review
topics_df = pd.DataFrame(with_docs, columns=[f"Topic {i}" for i in range(num_topics)])
print(topics_df)

################################################################################################



# Define general themes and their keywords
general_topics = {
    "Price": ["cheap", "expensive", "price", "value", "affordable", "cost-effective", "budget-friendly", "inexpensive", "pricy", "overpriced", "costly"],
    "Quality": ["good", "bad", "quality", "durable", "high-quality", "low-quality", "well-made", "fragile", "sturdy", "weak", "reliable", "unreliable"],
    "Use": ["easy", "difficult", "use", "works", "intuitive", "counterintuitive", "straightfoward", "complicated", "efficient", "inefficient", "unreliable"],
    "Design": ["nice", "ugly", "design", "aesthetic", "stylish", "unstylish", "atractive", "modern", "outdated", "elegant", "tastefull", "tasteless"],
}

with_docs = lda_model.transform(dtm)  

lda_topics = [[vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]] for topic in lda_model.components_]
mapped_topics = map_lda_to_general_topic(lda_topics, general_topics)
print(mapped_topics)

for i in range(len(sequences_list)):
    print(f"Sequence {i + 1}: {sequences_list[i]}")
    
    topic_probability = with_docs[i]
    assigned_topic = topic_probability.argmax()
    
    print(f"-- Assigned Topic: {assigned_topic} ({mapped_topics[f'Topic {assigned_topic}']})")
