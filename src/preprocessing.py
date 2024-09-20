import pandas as pd
from utils import detect_emojis, get_product_with_max_reviews, tokenize_reviews_to_sequences, get_product_with_n_reviews, lemmatisation_stopwords_series
import nltk

# get filepath to parsed file
processed_file_path = '../data/parsed_input_file.csv'

# turn the csv file to a panda's dataframe
dataset = pd.read_csv(processed_file_path)

# get the product with ten reviews
product_id = get_product_with_n_reviews(processed_file_path,20)

# get the review text for the product 10 reviews
reviews = (dataset[dataset['productId'] == product_id])['reviewText']

# detect emojis
if (detect_emojis(reviews)):
    print('There are emojis in the reviews')
else:
    print('There are no emojis in the reviews')


# lemmatisation and stopword removal
df_processed = lemmatisation_stopwords_series(reviews)

# tokenization in sequences of 5 words
sequences_list = tokenize_reviews_to_sequences(df_processed,5)

# vectorization
from sklearn.feature_extraction.text import CountVectorizer

# vectorizer = CountVectorizer(stop_words="english")
vectorizer = CountVectorizer()
dtm = vectorizer.fit_transform(sequences_list)

from sklearn.decomposition import LatentDirichletAllocation

# Apply LDA
num_topics = 5  # Set the number of topics you want
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


################################################################################################
def map_lda_to_general_topic(lda_topics, general_topics):
    mapped_topics = {}
    for idx, topic_words in enumerate(lda_topics):
        for general_topic, keywords in general_topics.items():
            if any(word in topic_words for word in keywords):
                mapped_topics[f"Topic {idx}"] = general_topic
                break
        else:
            mapped_topics[f"Topic {idx}"] = 'Other'
    return mapped_topics

# Define general themes and their keywords
general_topics = {
    "Price": ["cheap", "expensive", "price", "value"],
    "Quality": ["good", "bad", "quality", "durable"],
    "Use": ["easy", "difficult", "use", "works"],
    "Design": ["nice", "ugly", "design", "aesthetic"]
}

with_docs = lda_model.transform(dtm)  

lda_topics = [[vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]] for topic in lda_model.components_]
mapped_topics = map_lda_to_general_topic(lda_topics, general_topics)


for i in range(min(with_docs.shape[0], len(reviews))):
    print(f"Document {i + 1}:\n{reviews.iloc[i]}")
    
    topic_probability = with_docs[i]
    assigned_topic = topic_probability.argmax()
    
    print(f"-- Assigned Topic: {assigned_topic} ({mapped_topics[f'Topic {assigned_topic}']})")

    for j in range(len(mapped_topics)):
        print(f"-- Topic: {j} ({mapped_topics[f'Topic {j}']}) (Probability: {topic_probability[j]:.2%})")
    
    print()


