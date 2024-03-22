import batch_processing as bp
import evaluate as ev


# Specify the directory containing story files
directory = '/home/rajagopalmenon.v/cnn/stories'
num_sentences = 3
limit = 10000

# Process stories to get generated and reference summaries
generated_summaries, reference_summaries = bp.process_stories(directory, num_sentences, limit)

# Calculate ROUGE scores for the summaries
rouge_scores = ev.calculate_rouge_scores(generated_summaries, reference_summaries)
meteor_score = ev.calculate_meteor_scores(generated_summaries, reference_summaries)
bleu_score = ev.calculate_bleu_scores(generated_summaries, reference_summaries)

# Print the scores
print("ROUGE Scores:", rouge_scores)
print("METEOR Score:", meteor_score)
print("BLEU Score:", bleu_score)

# Print the first 5 generated summaries for demonstration
for summary in generated_summaries[:5]:
    print(summary)
    print('---')

for summary in reference_summaries[:5]:
    print(summary)
    print('---')
