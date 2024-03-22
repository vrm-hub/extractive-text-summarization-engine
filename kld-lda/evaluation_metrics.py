from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

def calculate_bleu(reference, generated):
    """
    Calculates BLEU score between a reference and a generated summary.

    Args:
    - reference (str): The reference summary.
    - generated (str): The generated summary.

    Returns:
    - float: The BLEU score.
    """
    return sentence_bleu([reference.split()], generated.split())

def calculate_rouge_meteor(reference, generated):
    """
    Calculates ROUGE scores and METEOR score between a reference and a generated summary.

    Args:
    - reference (str): The reference summary.
    - generated (str): The generated summary.

    Returns:
    - dict: A dictionary containing ROUGE-1, ROUGE-2, ROUGE-L scores.
    - float: The METEOR score.
    """
    rouge = Rouge()
    reference_tokens = reference.split()
    generated_tokens = generated.split()

    scores_rouge = rouge.get_scores(generated, reference, avg=True)
    meteor_score_value = meteor_score([reference_tokens], generated_tokens)

    return scores_rouge, meteor_score_value
