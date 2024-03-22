from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import torch
import numpy as np
from nltk.tokenize import word_tokenize


def evaluate_model(model, test_loader, tokenizer, device):
    """
    Evaluate the model on the test set.

    Args:
        model (torch.nn.Module): The trained model for summarization.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        tokenizer (Tokenizer): BERT or DistilBERT tokenizer.
        device (torch.device): The device to run the model on.

    Returns:
        tuple: Tuple containing lists of generated summaries and reference summaries.
    """
    model.eval()
    generated_summaries = []
    reference_summaries = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)

            # Here, outputs are assumed to be the sentence embeddings directly
            sentence_embeddings = outputs.mean(dim=1)

            predicted_summary = extract_summary_based_on_cosine_similarity(sentence_embeddings, batch['input_ids'],
                                                                           tokenizer)
            generated_summaries.append(predicted_summary)
            reference_summaries.append(batch['reference_summary'])

    return generated_summaries, reference_summaries


def extract_summary_based_on_cosine_similarity(embeddings, input_ids, tokenizer, top_k=3):
    """
    Extracts a summary based on cosine similarity.

    Args:
        embeddings (torch.Tensor): The sentence embeddings from the model.
        input_ids (torch.Tensor): The input IDs of the sentences.
        tokenizer (Tokenizer): BERT or DistilBERT tokenizer.
        top_k (int): Number of top sentences to include in the summary.

    Returns:
        str: The extracted summary.
    """
    # Convert embeddings to numpy array for cosine similarity computation
    embeddings_np = embeddings.cpu().numpy()

    # Reshape embeddings to 2D if necessary
    if embeddings_np.ndim == 1:
        embeddings_np = embeddings_np.reshape(-1, 1)

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings_np, embeddings_np)

    avg_similarity = similarity_matrix.mean(axis=1)

    top_sentence_indices = avg_similarity.argsort()[-top_k:][::-1]

    # Extract and decode the top sentences to form the summary
    summary_sentences = []
    for idx in top_sentence_indices:
        sentence = tokenizer.decode(input_ids[idx], skip_special_tokens=True)
        summary_sentences.append(sentence)

    summary = ' '.join(summary_sentences)
    return summary


def calculate_metrics(generated_summaries, reference_summaries):
    """
    Calculate ROUGE, BLEU, and METEOR scores for the summaries.

    Args:
        generated_summaries (list): List of generated summaries.
        reference_summaries (list): List of reference summaries.

    Returns:
        dict: Dictionary containing ROUGE, BLEU, and METEOR scores.
    """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)

    bleu_scores = [nltk.translate.bleu_score.sentence_bleu([word_tokenize(ref)], word_tokenize(gen)) for gen, ref in
                   zip(generated_summaries, reference_summaries)]
    avg_bleu_score = np.mean(bleu_scores)

    meteor_scores = [nltk.translate.meteor_score.single_meteor_score(word_tokenize(ref), word_tokenize(gen)) for
                     gen, ref in zip(generated_summaries, reference_summaries)]
    avg_meteor_score = np.mean(meteor_scores)

    return {
        'rouge': rouge_scores,
        'bleu': avg_bleu_score,
        'meteor': avg_meteor_score
    }


def flatten_summaries(summaries):
    """
    Flattens a list of summaries where each summary could be a list of sentences.
    Each summary is converted to a single string.

    Args:
        summaries (list): List of summaries, where each summary can be a list of sentences or a single string.

    Returns:
        list: List of flattened summary strings.
    """
    flattened_summaries = []
    for summary in summaries:
        if isinstance(summary, list):
            # Join list of sentences into a single string
            flattened_summary = ' '.join(summary)
            flattened_summaries.append(flattened_summary)
        elif isinstance(summary, str):
            flattened_summaries.append(summary)
        else:
            raise TypeError("Summary is neither a string nor a list of strings.")
    return flattened_summaries
