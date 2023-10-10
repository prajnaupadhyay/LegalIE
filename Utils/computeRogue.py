#!pip install rouge_score
from rouge_score import rouge_scorer
import numpy as np
import sys

def calculate_rouge_score(model, reference_file, prediction_file):
    test_labels = []
    predictions = []
    #REfernce file 
    with open(reference_file, 'r') as file:
        for line in file:
            if line.startswith('Prediction:') and line.strip() != "":
                line = line.strip("Prediction: ")
                if "COORDINATION(" in line:
                    line = line.replace("COORDINATION(", "")
                    line = line.replace(')', '')
                    test_labels.append(line)
                else:
                    test_labels.append(line)
                    
    #test_labesl = np.array(test_labels)
   
    pred_file = open(prediction_file, 'r')
    if model.lower() == "openie":
        predictions = pred_file.read().split('\n\n')
        for i in range(len(predictions)):
            _t = [s.strip() for s in predictions[i].splitlines() if s.strip() != ""][1:]
            predictions[i] = "\"\"".join(_t)
        predictions = predictions[:-1]
    else:
        with open(prediction_file, 'r') as file:
            for line in file:
                if line.startswith('Prediction:'):
                    line = line.strip("Prediction: ")
                    
                    if "COORDINATION(" in line:
                        line = line.replace("COORDINATION(", "")
                        line = line.replace("COORDINATIONAL(", '')
                        line = line.replace(')','')
                        #line = line.replace(',None','')
                    
                        predictions.append(line.strip())    
                    # if line.endswith(','):
                    #     predictions.append(line.strip()[-1])
                    else:
                        predictions.append(line.strip())
 
    
    print("Total number of test and predictions:",len(test_labels), len(predictions),"\n", test_labels[0], predictions[0])
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL']) #, use_stemmer=True

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




if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python computeRougue.py OpenIE/T5/BART <prediction_file> <result_file>")
        exit(0)
    reference_file_path = 'data/CoordinationDataSet/gold/test_copy.coord'
    # reference_file_path = sys.argv[1]
    prediction_file_path = sys.argv[2]


    rouge_scores = calculate_rouge_score(sys.argv[1], reference_file_path, prediction_file_path)
    result = open(sys.argv[3], "w")
    for metric, values in rouge_scores.items():
        print(f"{metric}:")
        print(f"  Precision: {values['p']:.4f}")
        print(f"  Recall: {values['r']:.4f}")
        print(f"  F1 Score: {values['f']:.4f}")
        
        result.write(f"{metric}:"+"\n\n")
        result.write(f"  Precision: {values['p']:.4f}"+"\n")
        result.write(f"  Recall: {values['r']:.4f}"+"\n")
        result.write(f"  F1 Score: {values['f']:.4f}"+"\n")
