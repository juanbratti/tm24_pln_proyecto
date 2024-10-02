from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

def apply_bertopic(sequences, seed_topic_list):
    """
    Apply BERTopic to extract topics from the reviews using seed topic list.

    Args:
        sequences (list): list/series of tokenized sequences 
        seed_topic_list (list): list of seed topics to guide the model.

    Returns:
        model (BERTopic): Trained BERTopic model.
        topics (list): list of topics found in the sequences.
    """

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = embedding_model.encode(sequences, show_progress_bar=True)

    topic_model = BERTopic(seed_topic_list=seed_topic_list)
    # topic_model = BERTopic()

    topic_model.fit_transform(sequences, embeddings)

    return topic_model
