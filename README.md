# Extractive Text Summarization Engine

## INTRODUCTION:
In our project, we focused on extractive text summarization in natural language processing. The goal was to
efficiently pull out important information from large amounts of text. This is particularly useful for tasks where
summarizing content can save time and enhance information retention, such as for students dealing with extensive
reading materials. We built and assessed three models for extractive summarization using datasets from Hermann
et al. (2015), derived from CNN news articles. These models included a pre-trained DistilBERT model fine-tuned
on CNN data, a BERT embeddings model with cosine similarity, and an LDA model using KL divergence.
## DATASET:
Our data consists of story files that begin with the main content followed by highlights (summaries) from CNN
news articles' authors. The datasets, crafted by Hermann et al. (2015) for Q&A research, are sourced from CNN
news articles. There are two datasets, one containing around 90k documents and the other approximately 197k
documents. We have used 10k stories from the set of 90k documents in this project. Each document consists of a
story followed by the abstractive summaries(highlights) of the story from the authors. The abstractive summaries
are later converted to extractive summaries by first identifying the most similar sentence from the original story
based on the overlap of words. The selected sentences are then combined to form an extractive summary,
capturing key information from the story. Link to the dataset: https://cs.nyu.edu/~kcho/DMQA/
## MODELS INVOLVED:
To perform extractive summarization, we utilized several models including pre-trained DistilBERT model that
underwent training on CNN data, a BERT embeddings model employing cosine similarity, and an LDA model
utilizing KL divergence.
Our approach refrains from extensive data preprocessing such as lemmatization, as applying such techniques
could compromise the inherent meaning of sentences. Preserving the original form of the text ensures that the
nuances and contextual information essential for extractive summarization are retained.
The 3 models are - 
### Cosine similarity using BERT Embeddings:
   Here's a step-by-step flow of the architecture:
1. Initialization and Setup (Top-Level Script):
- Import necessary libraries and modules.
- Initialize BERT tokenizer and model from the transformers library.
- Specify the device for computation (CPU or GPU).
2. Process Stories (batch_processing.py):
- Define a function `process_stories` that takes a directory containing story files as input.
- Load previously processed data from the `processed_data_dir` and skip those files.
- For each story file in the specified directory, do the following:
- Check if the story has already been processed; if yes, skip to the next one.
- Preprocess the story by tokenizing and converting it to extractive format.
- Tokenize and filter sentences based on length.
- Calculate BERT embeddings for the tokenized chunks of the story.
- Summarize the article using the `summarize_article` function.
- Save the processed data (story text, tokenized sentences, generated summary, reference summary) to a
  pickle file.
- Append the generated and reference summaries to lists for later evaluation.
3. Bert Embeddings (bert_embeddings.py):
- Define a function `get_sentence_embeddings` that generates sentence embeddings using a pre-trained
  BERT model.
- The function takes a text chunk, BERT tokenizer, BERT model, and computation device as input.
- Tokenize the input text, limit the sequence length to 512 tokens, and feed it through the BERT model.
- Extract and process the output to get sentence embeddings using mean pooling.
4. Evaluation (evaluate.py):
- Define functions to calculate ROUGE, METEOR, and BLEU scores for generated summaries against
  reference summaries.
- Use the `calculate_rouge_scores`, `calculate_meteor_scores`, and `calculate_bleu_scores` functions.
5. Demo Script (demo.py):
- Import the batch_processing and evaluate modules.
- Specify the directory containing story files, the number of sentences to include in each summary, and a
  limit on the number of files to process.
- Process stories to get generated and reference summaries.
6. Preprocessing (preprocessing.py):
- Define functions for preprocessing text and story files.
- Tokenize text into sentences, preprocess story files to extract story text and reference summaries, and
  convert abstractive to extractive summaries.
- These functions are used in the `process_stories` function.
7. Summarization (summarization.py):
- Define functions for calculating the cosine similarity matrix, scoring sentences based on embeddings,
  and generating a summary of an article.
- These functions are used in the `process_stories` function.
8. Utility (utility.py):
- Define functions for saving and loading data using pickle.
- The `save_data` and `load_data` functions are used to store and retrieve processed data.
  **Evaluation done with 5000 and 10000 datasets
### Training DistilBERT:
1. Data Preprocessing(data_processing.py):
- Import necessary libraries (torch, sklearn, nltk, etc.).
- Define a custom dataset (SummarizationDataset) for handling tokenized sentences, attention masks,
  scores, and reference summaries.
- Implement functions for filtering top sentences based on TF-IDF scores (filter_top_sentences), saving
  and loading data using pickle (save_data, load_data), and preprocessing text (preprocess_text).
- Load and preprocess data from a directory (load_and_process_data).
- Calculate relevance scores using cosine similarity (calculate_relevance_scores).
2. Model Architecture(model.py):
- Define a PyTorch model class (SummarizationModel) that uses the DistilBERT model for sentence
  embeddings.
- The model includes a linear layer for classification, assuming the output is pooled to get one value per
  sentence.
3. Model Training(train.py):
- Import necessary libraries (torch, optim, nn).
- Use the custom model (SummarizationModel) and train it using a specified training loader (train_model).
- The training process involves using the Cosine Embedding Loss and Adam optimizer.
4. Evaluation:
- Import libraries for evaluation (ROUGE, cosine_similarity, nltk).
- Define functions for evaluating the model (evaluate_model,
  extract_summary_based_on_cosine_similarity, calculate_metrics).
- Evaluate the trained model on the test set and calculate ROUGE, BLEU, and METEOR scores.
5. Final Evaluation:
- Load the trained model.
- Evaluate the model on the test set and calculate evaluation metrics.
- Print the ROUGE, BLEU, and METEOR scores.
  **Evaluation done with 5000 and 10000 datasets
  **Hyperparameters changed: epochs - 3, 5, 10
  batch size - 32, 64
### LDA Model with KL Divergence:
   During our initial phase of experimentation and exploration, we initially employed KL divergence for extractive
   text summarization. However, as our understanding evolved and we delved deeper into the intricacies of the task,
   we recognized the advantages of incorporating LDA (Latent Dirichlet Allocation) in conjunction with KL
   divergence. This realization emerged as a pivotal enhancement, as LDA introduces a probabilistic topic modeling
   aspect, enabling the identification of latent topics within the text. This, paired with KL divergence, facilitates a
   nuanced assessment of divergence in word probabilities. The model not only considers dissimilarities in word
   distributions but also comprehends broader contextual topics, resulting in a more effective and nuanced extraction
   of key information.
   Here's the step-by-step flow:
1. Data Preprocessing (data_preprocessing.py):
- Reads the contents of a given file.
- Splits the story into text and highlights using the '@highlight' delimiter.
- Cleans the story text by removing extra white spaces.
- Creates a list of highlights, forming the reference summary.
2. Abstractive to Extractive Conversion (abstractive_to_extractive.py):
- Tokenizes abstractive summary and story sentences using NLTK.
- Compares each abstractive sentence to all story sentences.
- Selects the story sentence with the highest overlap in terms of word tokens.
- Forms an extractive summary using the selected story sentences.
3. LDA-based Summarization (lda_summarization.py):
- Tokenizes the input document into sentences and words using NLTK.
- Creates a Gensim Dictionary and a bag-of-words corpus for the document.
- Applies Latent Dirichlet Allocation (LDA) model with 10 topics on the corpus.
- Extracts the word probability distributions for each topic.
- Iteratively selects sentences with the lowest Kullback-Leibler divergence from the LDA model's word
  distribution until the desired summary length is reached.
4. Evaluation Metrics (evaluation_metrics.py):
- Uses NLTK's BLEU score implementation to calculate BLEU score.
- Utilizes Rouge library for calculating ROUGE scores.
- Employs the Meteor library for calculating METEOR score.
5. Main File (main.py):
- Iterates over a set of files in a specified dataset folder.
- Reads each file's content, performs data preprocessing, and obtains the reference summary.
- Converts the abstractive summary to an extractive summary.
- Applies LDA-based summarization on the story text.
- Calculates BLEU, ROUGE, and METEOR scores between the reference summary and LDA-based
  summary.
- Keeps track of these scores for later averaging.
- Prints the story, reference summary, and LDA summary for each file.
- Finally, calculates and prints the average BLEU, ROUGE, and METEOR scores over all processed files.
## CHALLENGES:
1. Transition Challenges: The initial exploration with Seq2Seq LSTM models for abstractive
summarization revealed a misalignment with the goal of extractive summarization. This prompted a shift
towards alternative methods, including the investigation of KL Divergence.
2. Resource Constraints and Data Processing Bottleneck: Computational resources presented challenges
during both data processing and model training. Data preprocessing, taking 8-9 hours for a 10,000-
document dataset on a 64GB RAM machine, limited the ability to scale up to larger datasets. Training
the model was time-intensive, with each epoch requiring 2-3 hours, hindering rapid experimentation and
the exploration of diverse model configurations. Addressing these resource constraints is crucial for
efficient model development and experimentation.
## CONCLUSION:
The first model, Cosine similarity using BERT Embeddings performed well by creating summaries that closely
resembled the reference for most parts. It was the best among the three models, providing coherent and complete
sentences. Additionally, it required less training, making it suitable for extractive summarization, where we
extract important sentences rather than generating new ones.
On the other hand, retraining the DistilBERT model was challenging because it needed more data and
computational resources. The complexity of the model made training on a local machine difficult. Despite trying
various adjustments, we couldn't achieve results like the reference, leading to output misalignment.