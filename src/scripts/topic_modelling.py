from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

def apply_bertopic(sequences, seed_topic_list):
    """
    Apply BERTopic to extract topics from the reviews using seed topic list.

    Args:
        sequences (list): list/series of tokenized sequences 
        seed_topic_list (list): list of seed topics to guide the model.

    Returns:
        topic_model (BERTopic): Trained BERTopic model.
        topics (list): list of topics found in the sequences.
    """

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = embedding_model.encode(sequences, show_progress_bar=True)

    topic_model = BERTopic(seed_topic_list=seed_topic_list)
    # topic_model = BERTopic()

    topic_model.fit_transform(sequences, embeddings)
    
    topic_model.reduce_topics(sequences, nr_topics=10)

    #Probar el reduce outliers y merge topics

    return topic_model

def visualize(topic_model, sequences_list):
    fig = topic_model.visualize_topics()
    fig.write_image("./images/topics.png")
    fig = topic_model.visualize_hierarchy()
    fig.write_image("./images/topic_hierarchy.png")
    fig = topic_model.visualize_barchart()
    fig.write_image("./images/topic_barchart.png")
    fig = topic_model.visualize_heatmap()
    fig.write_image("./images/topic_similarity_heatmap.png")
    fig = topic_model.visualize_term_rank()
    fig.write_image("./images/term_score_decline.png")

    topic_distr, _ = topic_model.approximate_distribution(sequences_list)
    # Elegir el doc que uno quiera y muestra la distribucion de prob
    fig = topic_model.visualize_distribution(topic_distr[1])
    fig.write_image("./images/topic_dist.png")
    return