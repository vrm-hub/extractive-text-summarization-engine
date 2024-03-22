import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import os
import time
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class SummarizationDataset(Dataset):
    def __init__(self, tokenized_sentences, attention_masks, scores, reference_summaries):
        self.tokenized_sentences = tokenized_sentences
        self.attention_masks = attention_masks
        self.scores = scores
        self.reference_summaries = reference_summaries

    def __len__(self):
        return len(self.tokenized_sentences)

    def __getitem__(self, idx):
        return {
            'input_ids': self.tokenized_sentences[idx],
            'attention_mask': self.attention_masks[idx],
            'scores': self.scores[idx],
            'reference_summary': self.reference_summaries[idx]
        }


def filter_top_sentences(sentences, tokenized_chunks, attention_masks, scores, top_n=5):
    """
    Filters the top N sentences based on their relevance scores.

    Args:
    sentences (list): List of sentences.
    tokenized_chunks (list): Tokenized representations of sentences.
    attention_masks (list): Attention masks for tokenized sentences.
    scores (list): Relevance scores for each sentence.
    top_n (int): Number of top sentences to select.

    Returns:
    tuple: Filtered sentences, tokenized chunks, attention masks, and scores.
    """
    if not sentences:
        return [], [], [], []

    # Calculate TF-IDF scores for each sentence
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1).A1

    # Determine top N sentence indices
    top_indices = np.argsort(sentence_scores)[-top_n:]

    # Filter based on top indices
    filtered_sentences = []
    filtered_tokenized_chunks = []
    filtered_attention_masks = []
    filtered_scores = []

    for index in top_indices:
        filtered_sentences.append(sentences[index])
        filtered_tokenized_chunks.append(tokenized_chunks[index])
        filtered_attention_masks.append(attention_masks[index])
        filtered_scores.append(scores[index])

    return filtered_sentences, filtered_tokenized_chunks, filtered_attention_masks, filtered_scores


def save_data(tokenized_sentences, attention_masks, scores, reference_summaries, save_dir):
    """
    Saves tokenized sentences, attention masks, scores, and reference summaries to a specified directory using pickle.

    Args:
    tokenized_sentences (list): A list of tokenized sentences.
    attention_masks (list): A list of attention masks.
    scores (list): A list of relevance scores.
    reference_summaries (list): A list of reference summaries.
    save_dir (str): The directory where the data will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filenames = ['tokenized_sentences.pkl', 'attention_masks.pkl', 'scores.pkl', 'reference_summaries.pkl']
    data_list = [tokenized_sentences, attention_masks, scores, reference_summaries]

    for filename, data in zip(filenames, data_list):
        with open(os.path.join(save_dir, filename), 'wb') as file:
            pickle.dump(data, file)


def load_data(save_dir):
    """
    Loads tokenized sentences, attention masks, scores, and reference summaries from a specified directory.

    Args:
    save_dir (str): The directory from which the data will be loaded.

    Returns:
    tuple: A tuple containing lists of tokenized sentences, attention masks, scores, and reference summaries.
    """
    filenames = ['tokenized_sentences.pkl', 'attention_masks.pkl', 'scores.pkl', 'reference_summaries.pkl']
    loaded_data = []

    for filename in filenames:
        with open(os.path.join(save_dir, filename), 'rb') as file:
            data = pickle.load(file)
            loaded_data.append(data)

    return tuple(loaded_data)


def preprocess_text(text, tokenizer, device, max_length=512):
    """
    Preprocesses text by tokenizing and encoding each sentence for BERT processing.

    Args:
    text (str): The text to be preprocessed.
    tokenizer (BertTokenizer): Tokenizer for BERT.
    device (torch.device): Device for tensors.
    max_length (int): Maximum length for tokenization.

    Returns:
    tuple: A tuple containing lists of sentences, tokenized chunks, and attention masks.
    """
    sentences = sent_tokenize(text)
    tokenized_chunks = []
    attention_masks = []

    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=True)

        if len(tokenized_sentence) > max_length:
            tokenized_sentence = tokenized_sentence[:max_length - 1] + [tokenizer.sep_token_id]

        attention_mask = [1] * len(tokenized_sentence) + [0] * (max_length - len(tokenized_sentence))
        tokenized_chunks.append(torch.tensor(tokenized_sentence, device=device))
        attention_masks.append(torch.tensor(attention_mask, device=device))

    return sentences, tokenized_chunks, attention_masks


def load_and_process_data(tokenizer, device, directory, max_length=512, limit=10000, top_n=5):
    """
    Load and process data from a directory containing story files,
    converting each story into tokenized chunks and calculating relevance scores.

    Args:
    tokenizer (BertTokenizer): BERT tokenizer.
    device (torch.device): Device for tensors.
    directory (str): Directory containing story files.
    max_length (int): Maximum length for tokenization.
    limit (int): Maximum number of files to process.
    top_n (int): Number of top sentences to select for each story.

    Returns:
    None: Saves the processed data in a specified directory.
    """
    count = 0
    all_tokenized_chunks = []
    all_attention_masks = []
    all_scores = []
    all_reference_summaries = []

    for filename in os.listdir(directory):
        if count >= limit:
            break
        start = time.time()

        if filename.endswith('.story'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                story = file.read()
            story_text, highlights = story.split('@highlight', 1)
            story_text = story_text.strip()

            highlights_list = highlights.split('@highlight')
            reference_summary = ''
            for highlight in highlights_list:
                reference_summary += highlight.strip() + '. '

            sentences, tokenized_chunks, attention_masks = preprocess_text(story_text, tokenizer, device, max_length)
            if sentences:
                scores = calculate_relevance_scores(sentences, reference_summary)

                filtered_sentences, filtered_tokenized_chunks, filtered_attention_masks, filtered_scores \
                    = filter_top_sentences(sentences, tokenized_chunks, attention_masks, scores, top_n)

                all_tokenized_chunks.extend(filtered_tokenized_chunks)
                all_attention_masks.extend(filtered_attention_masks)
                all_scores.extend(filtered_scores)
                all_reference_summaries.append(reference_summary)
            else:
                print(f"No content to process for file: {filename}")
                continue

            print(f"Processed file {count} - {filename} - time taken: {time.time() - start}")
            count += 1

    processed_data_dir = '../processed-data-distilbert-7'
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    save_data(all_tokenized_chunks, all_attention_masks, all_scores, all_reference_summaries, processed_data_dir)


def calculate_relevance_scores(sentences, reference_summary):
    """
    Calculate relevance scores for each sentence based on their overlap with the reference summary.

    Args:
        sentences (list): List of sentences from the article.
        reference_summary (str): The reference summary.

    Returns:
        list: List of relevance scores for each sentence.
    """
    if not sentences:
        return []

    documents = sentences + [reference_summary]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
    similarities = cosine_similarities.flatten()

    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    normalized_scores = (similarities - min_sim) / (max_sim - min_sim) if max_sim - min_sim else similarities

    return normalized_scores.tolist()


def create_data_loaders(tokenized_sentences, attention_masks, scores, reference_summaries, batch_size):
    """
    Creates data loaders for training, validation, and testing.

    Args:
        tokenized_sentences (list): List of tokenized sentences.
        attention_masks (list): List of attention masks.
        scores (list): List of scores for each sentence.
        reference_summaries (list): List of reference summaries.
        batch_size (int): Batch size for data loaders.

    Returns:
        tuple: Tuple of data loaders for training, validation, and testing.
    """
    data = list(zip(tokenized_sentences, attention_masks, scores, reference_summaries))
    print(f"Total dataset size: {len(data)}")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

    print(f"Train dataset size: {len(train_data)}")
    print(f"Validation dataset size: {len(val_data)}")
    print(f"Test dataset size: {len(test_data)}")

    train_dataset = SummarizationDataset(*zip(*train_data))
    val_dataset = SummarizationDataset(*zip(*val_data))
    test_dataset = SummarizationDataset(*zip(*test_data))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Number of batches in train_loader: {len(train_loader)}")

    return train_loader, val_loader, test_loader
