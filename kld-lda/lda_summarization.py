from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np

def lda_sum(document, summary_length):
    """
        Generates a summary using Latent Dirichlet Allocation (LDA) model.

        Args:
        - document (str): The input document to be summarized.
        - summary_length (int): The desired length of the summary in sentences.

        Returns:
        - str: The generated summary.
    """
    sentences = sent_tokenize(document)

    words_in_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words_in_sentences.append(words)

    dictionary = Dictionary(words_in_sentences)

    corpus = []
    for words in words_in_sentences:
        bow = dictionary.doc2bow(words)
        corpus.append(bow)

    lda = LdaModel(corpus, num_topics=20, id2word=dictionary)

    topics = lda.show_topics(formatted=False)
    word_pd = {}
    for topic in topics:
        for word, prob in topic[1]:
            word_pd[word] = prob

    summary = []
    divergence = float('inf')

    while len(summary) < summary_length and sentences:
        min_divergence = divergence
        min_sentence = None
        for sentence in sentences:
            sentence_word_freq = Counter(word_tokenize(sentence))
            sentence_word_pd = {word: freq / sum(sentence_word_freq.values()) for word, freq in sentence_word_freq.items()}
            kl_divergence = sum(word_pd[word] * np.log(word_pd[word] / sentence_word_pd.get(word, 1e-8)) for word in word_pd)
            if kl_divergence < min_divergence:
                min_divergence = kl_divergence
                min_sentence = sentence

        summary.append(min_sentence)
        sentences.remove(min_sentence)

    return ' '.join(summary)
