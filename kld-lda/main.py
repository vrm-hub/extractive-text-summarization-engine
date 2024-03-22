import os
import random
from data_preprocessing import preprocess_data
from abstractive_to_extractive import convert_abstractive_to_extractive
from lda_summarization import lda_sum
from evaluation_metrics import calculate_bleu, calculate_rouge_meteor

dataset_folder = "/Users/harshithaprabhu/Downloads/cs6120 final project/stories"

all_files = []
for file_name in os.listdir(dataset_folder):
    if file_name.endswith(".story"):
        all_files.append(file_name)

random.seed(42)
selected_files = all_files[:5000]

bleu_scores = []
rouge_scores_list = []
meteor_scores = []
count = 0

for file_name in selected_files:
    count += 1
    file_path = os.path.join(dataset_folder, file_name)
    story_text, reference_summary = preprocess_data(file_path)
    summary_length = 2
    if len(story_text.strip()) == 0:
        continue

    reference_summary = convert_abstractive_to_extractive(reference_summary, story_text)
    lda_summary = lda_sum(story_text, summary_length)

    print(f"\nStory:\n{story_text}\n")
    print(f"Reference Summary:\n{reference_summary}\n")
    print(f"LDA Summary:\n{lda_summary}\n")

    bleu_score = calculate_bleu(reference_summary, lda_summary)
    rouge_scores, meteor_score_value = calculate_rouge_meteor(reference_summary, lda_summary)
    bleu_scores.append(bleu_score)
    rouge_scores_list.append(rouge_scores)
    meteor_scores.append(meteor_score_value)

avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
cumulative_rouge_scores = {metric: 0 for metric in rouge_scores_list[0]}
for rouge_scores in rouge_scores_list:
    for metric in rouge_scores:
        cumulative_rouge_scores[metric] += rouge_scores[metric]['f']

avg_rouge_scores = {}
for metric in rouge_scores_list[0]:
    avg_rouge_scores[metric] = cumulative_rouge_scores[metric] / len(rouge_scores_list)

avg_meteor_score = sum(meteor_scores) / len(meteor_scores)

print(f"\nAverage BLEU Score: {avg_bleu_score}")
print(f"Average ROUGE Scores: {avg_rouge_scores}")
print(f"Average METEOR Score: {avg_meteor_score}")
