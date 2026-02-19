"""
Lemma Tokenizer Module
----------------------
Implements the unified preprocessing pipeline.

Steps:
1. Tokenization (NLTK word_tokenize)
2. Stopword removal
3. POS tagging
4. Lemmatization
5. Abbreviation normalization

Compatible with Spark UDF usage.
"""

import re
import nltk
from typing import List
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

# Ensure resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

ABBREVIATIONS = {
    "ai": "artificial_intelligence",
    "ml": "machine_learning",
    "cc": "cloud_computing"
}


def normalize_abbreviations(token: str) -> str:
    """Replace known abbreviations."""
    return ABBREVIATIONS.get(token.lower(), token)


def get_wordnet_pos(treebank_tag: str) -> str:
    """Map POS tag to WordNet format."""
    if treebank_tag.startswith("J"):
        return "a"
    if treebank_tag.startswith("V"):
        return "v"
    if treebank_tag.startswith("N"):
        return "n"
    if treebank_tag.startswith("R"):
        return "r"
    return "n"


def lemma_tokenize(text: str) -> List[str]:
    """
    Full Lemma Tokenizer pipeline.

    Parameters
    ----------
    text : str
        Raw tweet text

    Returns
    -------
    List[str]
        Clean normalized tokens
    """

    if not text:
        return []

    # Clean URLs and mentions
    text = re.sub(r"http\S+|@\w+|#\w+", " ", text.lower())

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords + normalize abbreviations
    tokens = [
        normalize_abbreviations(t)
        for t in tokens
        if t.isalpha() and t not in STOP_WORDS
    ]

    # POS tagging
    tagged = pos_tag(tokens)

    # Lemmatization
    lemmas = [
        LEMMATIZER.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in tagged
    ]

    return lemmas
