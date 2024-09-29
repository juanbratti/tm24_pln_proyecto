import pandas as pd
import spacy
import nltk
from nltk import ngrams
from utils import *
import re

nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

def preprocess(text, onlyadj, nonouns, nostopwords):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if onlyadj:
            if token.pos_ == "ADJ":
                tokens.append(token.lemma_)
        if nonouns:
            if token.pos_ != "NOUN":
                if nostopwords:
                    if not token.is_stop:
                        tokens.append(token.lemma_)
                else:
                    tokens.append(token.lemma_)
        if nostopwords:
            if not token.is_stop:
                tokens.append(token.lemma_)

    return " ".join(tokens)

# Función para crear n-gramas
def create_ngrams(text, n):
    tokens = nltk.word_tokenize(text)  # Tokenizar el texto
    return [' '.join(gram) for gram in ngrams(tokens, n)]  # Crear n-gramas




# Función para dividir texto por puntos o comas
def split_by_punctuation(text):
    return re.split(r'[.,]', text)
