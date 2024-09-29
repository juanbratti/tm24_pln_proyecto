from kmeans_test import *
import pandas as pd
from utils import *
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Variables
# Definir los valores de estas variables y luego cual es el dataset de reviews
min_reviews_per_product = 30
max_reviews_per_product = 40
amount_of_products = 1
onlyadj = False # si se quiere solo los adjetivos de las reviews
nonouns = True # si se quiere sacar los sustantivosde las reviews
nostopwords = True # si se quiere sacar las stopwords
n_grams = True # n secuencias de palabras
n = 5
punctuation = False # si se prefiere separar por puntos y comas
n_clusters = 5 # cantidad de clusters
show = False # se muestra el grafico
savefig = True # se guarda el grafico
important_words = 5 # cantidad de palabras mas importantes del cluster imprimidas



# Definir las reviews
processed_file_path = '../data/parsed_input_file.csv'

product_ids = get_products_with_reviews_in_range(processed_file_path,
                                         amount_of_products,
                                         30, 40)
reviews = []
dataset = pd.read_csv(processed_file_path)
for product_id in product_ids: 
    reviews.extend(dataset[dataset['productId'] == product_id]['reviewText'].tolist())

# O definimos uno propio

# reviews = [
#     "This product is amazing! I love the quality and design.",
#     "Not worth the price. It broke after a few uses.",
#     "The best purchase I’ve made this year. Highly recommend!",
#     "Average product, nothing special but it gets the job done.",
#     "Terrible customer service. I would not buy from this seller again."
# ]

split_reviews = []
for review in reviews:
    processed_review = preprocess(review, onlyadj, nonouns, nostopwords)
    if punctuation:
        split_sentences = split_by_punctuation(processed_review)
        split_reviews.extend([sentence.strip() for sentence in split_sentences if sentence.strip()])
    if n_grams:
        ngrams_for_review = create_ngrams(processed_review, n)
        split_reviews.extend(ngrams_for_review)

# Vectorización con TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(split_reviews)

# Aplicar K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_
print(f'Clusters: {labels}')

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1] 
terms = tfidf.get_feature_names_out() 

for i in range(n_clusters):
    print(f"\nCluster {i}:")
    total_weight = kmeans.cluster_centers_[i].sum()
    for ind in order_centroids[i, :important_words]:
        weight = kmeans.cluster_centers_[i, ind]
        percentage = (weight / total_weight) * 100
        print(f'{terms[ind]}: {percentage:.2f}%')


pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())


# Gráfico de dispersión con centroides
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', marker='o')
centroids = kmeans.cluster_centers_
centroids_reduced = pca.transform(centroids)
plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('K-means Clusters with Centroids (PCA) - N-grams')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.legend()
if show:
    plt.show()
if savefig:
    plt.savefig('kmeans_clusters_centroids.png')
plt.close()