import os
import time
import preprocessing as pp
import bert_embeddings as be
import summarization as sm
import utility as ut
from transformers import BertTokenizer, BertModel
import torch

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model.to(device)


def process_stories(directory, num_sentences=3, limit=10000):
    """
    Processes stories from the given directory to generate summaries.
    It skips files that have been already processed and saves processed data to a separate directory.

    Args:
    directory (str): Directory containing story files.
    num_sentences (int): Number of sentences to include in each summary.
    limit (int): Maximum number of files to process.

    Returns:
    tuple: Tuple containing two lists - generated summaries and reference summaries.
    """
    generated_summaries = []
    reference_summaries = []
    count = 0

    # Directory for saving processed data
    processed_data_dir = '../processed_data-2-1'
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    # Load already processed files
    for filename in os.listdir(processed_data_dir):
        if filename.endswith('.pkl'):
            data = ut.load_data(os.path.join(processed_data_dir, filename))
            generated_summaries.append(data['generated_summary'])
            reference_summaries.append(data['reference_summary'])
            print(f"Processed file {count} - {filename}")
            count += 1

    # Process new files
    for filename in os.listdir(directory):
        start = time.time()
        if filename.endswith('.story') and count < limit:
            processed_data_file = os.path.join(processed_data_dir, f'{filename}_processed.pkl')

            # Skip already processed stories
            if os.path.exists(processed_data_file):
                continue

            # Process the story
            story_path = os.path.join(directory, filename)
            story_text, reference_summary = pp.preprocess_story_file(story_path)
            reference_summary = pp.convert_abstractive_to_extractive(reference_summary, story_text)
            sentences, tokenized_chunks = pp.preprocess_text(story_text, tokenizer, device)

            # Filter sentences with more than 3 words
            filtered_sentences = []
            for sentence in sentences:
                if len(sentence.split()) > 3:
                    filtered_sentences.append(sentence)
            if len(filtered_sentences) != len(sentences):
                continue

            if not story_text.strip():
                print(f"Empty story text for file: {filename}")
                continue

            all_embeddings = []
            for chunk in tokenized_chunks:
                embeddings = be.get_sentence_embeddings(chunk, model, device)
                for embedding in embeddings:
                    all_embeddings.append(embedding)

            # Check if both sentences and embeddings are empty
            if not sentences and not all_embeddings:
                print(f"No content to summarize for file: {filename}")
                continue

            print(f"Number of sentences: {len(sentences)}, Number of embeddings: {len(all_embeddings)}")

            summary = sm.summarize_article(sentences, all_embeddings, len(reference_summary.split(". ")))

            # Save the summary and reference summary
            save_data = {
                'story_text': story_text,
                'tokenized_sentences': sentences,
                'generated_summary': summary,
                'reference_summary': reference_summary
            }
            ut.save_data(save_data, processed_data_file)

            generated_summaries.append(summary)
            reference_summaries.append(reference_summary)

            print(f"Processed file {count} - {filename} - time taken: {time.time() - start}")
            count += 1

    return generated_summaries, reference_summaries