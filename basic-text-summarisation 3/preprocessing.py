import nltk
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import torch

nltk.download('punkt')


def convert_abstractive_to_extractive(abstractive_summary, story_text):
    """
    Converts an abstractive summary to an extractive summary by selecting sentences from the story text.

    Args:
    abstractive_summary (str): The abstractive summary of the story.
    story_text (str): The full text of the story.

    Returns:
        str: The extractive summary generated from the story text.
    """
    # Tokenize abstractive summary into sentences
    abstractive_sentences = sent_tokenize(abstractive_summary)

    # Tokenize story text into sentences
    story_sentences = sent_tokenize(story_text)

    extractive_summary = []
    for abs_sentence in abstractive_sentences:
        best_match = max(story_sentences,
                         key=lambda sentence: len(set(word_tokenize(abs_sentence)) & set(word_tokenize(sentence))),
                         default='')
        extractive_summary.append(best_match)

    return ' '.join(extractive_summary)


def preprocess_text(text, tokenizer, device, max_length=512):
    """
    Tokenizes text into segments suitable for BERT processing, with each segment
    containing a single sentence.

    Args:
    text (str): The text to be tokenized.
    max_length (int): Maximum length of a tokenized chunk.

    Returns:
    tuple: A tuple containing two elements:
        - List of sentences from the text.
        - List of tokenized chunks, each representing a single sentence as a PyTorch tensor.
        - List of attention masks corresponding to the tokenized chunks.
    """
    sentences = sent_tokenize(text)
    tokenized_chunks = []

    for sentence in sentences:
        # Tokenize each sentence individually with special tokens
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=True)

        # Truncate the sentence if it's too long
        if len(tokenized_sentence) > max_length:
            tokenized_sentence = tokenized_sentence[:max_length - 1] + [tokenizer.sep_token_id]

        # Convert to tensor and add to the list
        tokenized_chunks.append(torch.tensor(tokenized_sentence, device=device))

    return sentences, tokenized_chunks


def preprocess_story_file(file_path):
    """
    Reads a story file and extracts the story text and reference summary.

    Args:
    file_path (str): Path to the story file.

    Returns:
    tuple: A tuple containing:
        - The story text.
        - The reference summary extracted from highlights.
        - BERT tokenizer, model, and computation device.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        story = file.read()
    story_text, highlights = story.split('@highlight', 1)
    story_text = story_text.strip()

    highlights_list = highlights.split('@highlight')
    reference_summary = ''
    for highlight in highlights_list:
        reference_summary += highlight.strip() + '. '

    return story_text, reference_summary.strip()
