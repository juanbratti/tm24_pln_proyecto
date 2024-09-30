import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from utils import *
from kmeans_test import preprocess, split_by_punctuation


onlyadj = False # si se quiere solo los adjetivos de las reviews
nonouns = True # si se quiere sacar los sustantivosde las reviews
nostopwords = True # si se quiere sacar las stopwords
n_grams = False # n secuencias de palabras
n = 5
punctuation = True # si se prefiere separar por puntos y comas
important_words = 5 # cantidad de palabras mas importantes del cluster imprimidas


processed_file_path = '../data/parsed_input_file.csv'


dataset = pd.read_csv(processed_file_path)


product_id = get_product_with_n_reviews(processed_file_path, 40)


reviews = (dataset[dataset['productId'] == product_id])['reviewText']


if detect_emojis(reviews):
    print('There are emojis in the reviews')
else:
    print('There are no emojis in the reviews')


reviews_sentences = split_into_sentences(reviews)

split_reviews = []
for review in reviews:
    processed_review = preprocess(review, onlyadj, nonouns, nostopwords)
    if punctuation:
        split_sentences = split_by_punctuation(processed_review)
        split_reviews.extend([sentence.strip() for sentence in split_sentences if sentence.strip()])
    if n_grams:
        ngrams_for_review = create_ngrams(processed_review, n)
        split_reviews.extend(ngrams_for_review)


####################### BERTopic #######################

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embedding_model.encode(split_reviews, show_progress_bar=True)


seed_topic_list = [["price", "cheap", "expensive", "price", "value", "affordable", "cost-effective", "budget-friendly", "inexpensive", "pricy", "overpriced", "costly"],
["quality", "good", "bad", "quality", "durable", "high-quality", "low-quality", "well-made", "fragile", "sturdy", "weak", "reliable", "unreliable"],
["use", "easy", "difficult", "use", "works", "intuitive", "counterintuitive", "straightfoward", "complicated", "efficient", "inefficient", "unreliable"],
["design", "nice", "ugly", "design", "aesthetic", "stylish", "unstylish", "atractive", "modern", "outdated", "elegant", "tastefull", "tasteless"]]

topic_model = BERTopic(seed_topic_list=seed_topic_list)
# topic_model = BERTopic()

topics, probabilities = topic_model.fit_transform(split_reviews, embeddings)


print("Product Title:")
print(dataset[dataset['productId'] == product_id]['productTitle'].iloc[0])


print("Topics found via BERTopic:")
topic_info = topic_model.get_topic_info()
print(topic_info)


for topic in range(len(set(topics))):
    print(f"Topic {topic}:")
    print(topic_model.get_topic(topic), "\n")


for i, sequence in enumerate(split_reviews):
    print(f"Sequence {i + 1}: {sequence}")
    assigned_topic = topics[i]
    topic_probability = probabilities[i]
    print(f"-- Assigned Topic: {assigned_topic}")
    print(f"-- Topic Probability: {topic_probability}\n")
