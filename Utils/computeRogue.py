from rouge_score import rouge_scorer

def calculate_rouge_score(reference_file, prediction_file):
    test_labels = []
    predictions = []
    
    with open(reference_file, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                test_labels.append(line)
                #print(line)
    test_labels = np.array(test_labels)
    
    with open(prediction_file, 'r') as file:
        for line in file:
            if line.startswith('Prediction'):
                line = line.strip("Prediction:")
                predictions.append(line)
    predictions = np.array(predictions)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge_scores = {
        'rouge1': {'f': 0, 'p': 0, 'r': 0},
        'rouge2': {'f': 0, 'p': 0, 'r': 0},
        'rougeL': {'f': 0, 'p': 0, 'r': 0}
    }

    for ref, pred in zip(test_labels, predictions):
        scores = scorer.score(ref.strip(), pred.strip())
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            rouge_scores[metric]['f'] += scores[metric].fmeasure
            rouge_scores[metric]['p'] += scores[metric].precision
            rouge_scores[metric]['r'] += scores[metric].recall

    num_examples = len(test_labels)
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        rouge_scores[metric]['f'] /= num_examples
        rouge_scores[metric]['p'] /= num_examples
        rouge_scores[metric]['r'] /= num_examples

    return rouge_scores
reference_file_path = 'ptb_test.txt'
prediction_file_path = 'prediction_ptb_Level.txt'
test_labels = []
predictions = []

rouge_scores = calculate_rouge_score(reference_file_path, prediction_file_path)

for metric, values in rouge_scores.items():
    print(f"{metric}:")
    print(f"  Precision: {values['p']:.4f}")
    print(f"  Recall: {values['r']:.4f}")
    print(f"  F1 Score: {values['f']:.4f}")
#results obtained
# rouge1:
#   Precision: 0.8734
#   Recall: 0.8051
#   F1 Score: 0.8276
# rouge2:
#   Precision: 0.6471
#   Recall: 0.5874
#   F1 Score: 0.6076
# rougeL:
#   Precision: 0.8634
#   Recall: 0.7957
#   F1 Score: 0.8183
