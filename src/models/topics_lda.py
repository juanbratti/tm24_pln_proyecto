from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def apply_lda(sequences_list, num_topics):
    """
    Apply LDA to extract topics from the reviews.

    Args:
        sequences_list (list): list/series of tokenized sequences 
        
    Returns:
        topic_model (LDA): Trained LDA model.
    """

    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(sequences_list)

    # Apply LDA
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda_model.fit(dtm)

    return lda_model

def print_model_info_lda(lda_model, sequences_list, shown_docs):
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(sequences_list)

    # Print the topics found by the LDA model
    print("Topics found via LDA (in training):")
    for i, topic in enumerate(lda_model.components_):
        print(f"Topic {i}:")
        print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
        print("\n")

    with_docs = lda_model.transform(dtm)  

    lda_topics = [[vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]] for topic in lda_model.components_]
    mapped_topics = map_lda_to_general_topic(lda_topics, general_topics)


    for i in range(shown_docs):
        print(f"Sequence {i + 1}: {sequences_list[i]}")
        
        topic_probability = with_docs[i]
        assigned_topic = topic_probability.argmax()
        
        print(f"-- Assigned Topic: {assigned_topic} ({mapped_topics[f'Topic {assigned_topic}']})")

        for j in range(len(mapped_topics)):
            print(f"-- Topic: {j} ({mapped_topics[f'Topic {j}']}) (Probability: {topic_probability[j]:.2%})")
        
        print()
    return

################################################################################################
def map_lda_to_general_topic(lda_topics, general_topics):
    mapped_topics = {}
    for idx, topic_words in enumerate(lda_topics):
        for general_topic, keywords in general_topics.items():
            if any(any(keyword in word for word in topic_words) for keyword in keywords):
                mapped_topics[f"Topic {idx}"] = general_topic
                break
        else:
            mapped_topics[f"Topic {idx}"] = 'Other'
    return mapped_topics

# Define general themes and their keywords
general_topics = {
    "Price": ["cheap", "expensive", "price", "value", "affordable", "cost-effective", "budget-friendly", "inexpensive", "pricy", "overpriced", "costly"],
    "Quality": ["good", "bad", "quality", "durable", "high-quality", "low-quality", "well-made", "fragile", "sturdy", "weak", "reliable", "unreliable"],
    "Use": ["easy", "difficult", "use", "works", "intuitive", "counterintuitive", "straightfoward", "complicated", "efficient", "inefficient", "unreliable"],
    "Design": ["nice", "ugly", "design", "aesthetic", "stylish", "unstylish", "atractive", "modern", "outdated", "elegant", "tastefull", "tasteless"]
}