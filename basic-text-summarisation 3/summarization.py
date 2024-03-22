import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity_matrix(embeddings):
    """
    Calculates the cosine similarity matrix for a set of embeddings.

    Args:
    embeddings (list): A list of sentence embeddings.

    Returns:
    numpy.ndarray: A cosine similarity matrix for the given embeddings.
    """
    return cosine_similarity(embeddings)


def score_sentences(embeddings, sentences, similarity_matrix, diversity_factor=0.7):
    """
    Scores sentences based on embeddings and similarity matrix.

    Args:
    embeddings (list): A list of sentence embeddings.
    sentences (list): A list of sentences corresponding to the embeddings.
    similarity_matrix (numpy.ndarray): Cosine similarity matrix of embeddings.
    diversity_factor (float): Factor to adjust diversity in scoring.

    Returns:
    numpy.ndarray: An array of sentence scores.
    """
    scores = np.zeros(len(sentences))
    for i, embedding in enumerate(embeddings):
        scores[i] = embedding.sum()

    n = len(sentences)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                scores[i] += similarity_matrix[i][j] * diversity_factor

    return scores


def summarize_article(sentences, embeddings, num_sentences=3):
    """
    Generates a summary of an article.

    Args:
    sentences (list): A list of sentences from the article.
    embeddings (list): A list of sentence embeddings.
    num_sentences (int): Number of sentences to include in the summary.

    Returns:
    str: A summary of the article.
    """
    similarity_matrix = calculate_similarity_matrix(embeddings)
    scores = score_sentences(embeddings, sentences, similarity_matrix)

    ranked_sentences = []
    for i, score in enumerate(scores):
        ranked_sentences.append((score, sentences[i]))
    ranked_sentences.sort(reverse=True)

    selected_sentences = []
    for _, sentence in ranked_sentences[:num_sentences]:
        selected_sentences.append(sentence)

    return ' '.join(selected_sentences)
