from rouge import Rouge
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu


def calculate_rouge_scores(gen_summaries, ref_summaries):
    """
    Calculates ROUGE scores for a set of generated summaries against reference summaries.

    Args:
    gen_summaries (list): A list of generated summaries.
    ref_summaries (list): A list of reference summaries.

    Returns:
    dict: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    rouge = Rouge()
    scores = rouge.get_scores(gen_summaries, ref_summaries, avg=True)
    return scores


def calculate_meteor_scores(gen_summaries, ref_summaries):
    """
    Calculates METEOR scores for a set of generated summaries against reference summaries.

    Args:
    gen_summaries (list): A list of generated summaries.
    ref_summaries (list): A list of reference summaries.

    Returns:
    float: The average METEOR score.
    """
    tokenized_gen_summaries = []
    for summary in gen_summaries:
        tokenized_gen_summaries.append(nltk.word_tokenize(summary))

    tokenized_ref_summaries = []
    for summary in ref_summaries:
        tokenized_ref_summaries.append([nltk.word_tokenize(summary)])

    meteor_scores = []
    for gen, ref in zip(tokenized_gen_summaries, tokenized_ref_summaries):
        score = meteor_score(ref, gen)
        meteor_scores.append(score)

    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    return avg_meteor_score


def calculate_bleu_scores(gen_summaries, ref_summaries):
    """
    Calculates BLEU scores for a set of generated summaries against reference summaries.

    Args:
    gen_summaries (list): A list of generated summaries.
    ref_summaries (list): A list of reference summaries.

    Returns:
    float: The average BLEU score.
    """
    references = []
    for ref in ref_summaries:
        references.append([ref])

    bleu_score = corpus_bleu(references, gen_summaries)
    return bleu_score
