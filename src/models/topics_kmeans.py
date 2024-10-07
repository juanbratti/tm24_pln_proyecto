import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def apply_kmeans(sequences, n_clusters):
    """
    Apply Kmeans to extract topics from the reviews.

    Args:
        sequences (list): list/series of tokenized sequences 
        n_clusters (number): number of clusters
    Returns:
        topic_model (Kmeans): Trained Kmeans model.
        topics (list): list of topics found in the sequences.
    """
    tfidf = TfidfVectorizer()
    vect = tfidf.fit_transform(sequences)

    # Aplicar K-means
    topic_model = KMeans(n_clusters=n_clusters, random_state=42)
    topic_model.fit(vect)

    topics = topic_model.labels_

    return topic_model, topics

def visualize_kmeans(topic_model, sequences_list):
    tfidf = TfidfVectorizer()
    vect = tfidf.fit_transform(sequences_list)
    pca = PCA(n_components=2)
    vect_reduced = pca.fit_transform(vect.toarray())
    labels = topic_model.labels_
    plt.figure(figsize=(10, 7))
    plt.scatter(vect_reduced[:, 0], vect_reduced[:, 1], c=labels, cmap='viridis', marker='o')
    centroids = topic_model.cluster_centers_
    centroids_reduced = pca.transform(centroids)
    plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.title('K-means Clusters with Centroids (PCA) - N-grams')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.legend()
    plt.savefig('./images/kmeans_clusters_centroids.png')
    plt.close()
    return

def print_model_info_kmeans(topic_model, sequences_list, n_clusters, important_words):
    print("-----------------------------------------")
    print("Topics discovered Kmeans:")
    labels = topic_model.labels_
    print(f'Clusters: {labels}')
    tfidf = TfidfVectorizer()
    vect = tfidf.fit_transform(sequences_list)
    order_centroids = topic_model.cluster_centers_.argsort()[:, ::-1] 
    terms = tfidf.get_feature_names_out() 

    for i in range(n_clusters):
        print(f"\nCluster {i}:")
        total_weight = topic_model.cluster_centers_[i].sum()
        for ind in order_centroids[i, :important_words]:
            weight = topic_model.cluster_centers_[i, ind]
            percentage = (weight / total_weight) * 100
            print(f'{terms[ind]}: {percentage:.2f}%')
    print("-----------------------------------------")
    return