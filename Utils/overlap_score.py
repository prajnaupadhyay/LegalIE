import numpy as np
import sys
from .wire57 import wire57_scorer

def get_sentences_from_tree_labels(model = 'T5', prediction = None):
    if tree_label == "NONE":
        return [""]
    count = tree_label.count("COORDINATION")
    count += tree_label.count("CO\\")
    count += tree_label.count("SUB\\")
    # Removing " )) from the end
    if count >= 1:
        tree_label = tree_label[:-(count + 2)]
    # if(model == "OpenIE"):
    #     sentences = tree_label.split("\" , \"")
    # else:
    #     sentences = tree_label.split("\", \"")
    sentences = tree_label.split("\",\"")
    new_sentenes = []
    if model in ["T5", "OpenIE", "BART"]:
        for s in sentences:
            if "\" COORDINATION" in s:
                s1 = s.split("\" COORDINATION(\"")
                for ss in s1:
                    ss = ss.replace("COORDINATION(\" ", "")
                    new_sentenes.append(ss)
            elif "COORDINATION" in s:
                s1 = s.split("COORDINATION(\"")
                for ss in s1:
                    ss = ss.replace("COORDINATION(\"", "")
                    new_sentenes.append(ss)
            else:
                s = s.replace("COORDINATION(\" ", "")
                new_sentenes.append(s)
    else:
        print("Invalid model name")
        sys.exit(0)
    return new_sentenes
    

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    targets_list = []  # Initialize lists to store targets and predictions
    predictions_list = []

    target = []  # Initialize variables to store individual targets and predictions
    prediction = []

    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            if target:
                targets_list.append(target)
            target = [line]  # Start a new target
        elif line.startswith("Prediction:"):
            if prediction:
                predictions_list.append(prediction)
            prediction = [line]  # Start a new prediction
        else:
            if target:
                target.append(line)  # Append lines to the current target
            if prediction:
                prediction.append(line)  # Append lines to the current prediction

    if target:
        targets_list.append(target)  # Append the last target if it exists
    if prediction:
        predictions_list.append(prediction)  # Append the last prediction if it exists

    return targets_list, predictions_list  # Return the lists of targets and predictions

def calculate_precision_recall_f1(targets_list, predictions_list):
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    precision_list = []
    recall_list = []
    f1_score_list = []

    for targets, predictions in zip(targets_list, predictions_list):
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for target, prediction in zip(targets, predictions):
            target_words = set(target.split()[1:-1])
            prediction_words = set(prediction.split()[1:-1])
            target_labels = set([word for word in target.split() if word.startswith(("CO/", "SUB/", "COORDINATION"))])
            prediction_labels = set([word for word in prediction.split() if word.startswith(("CO/", "SUB/", "COORDINATION"))])

            # Calculate overlap (excluding labels)
            overlap = len(target_words.intersection(prediction_words))

            # Calculate label overlap
            label_overlap = len(target_labels.intersection(prediction_labels))

            # Calculate true positives false positives, and false negatives
            true_positives += (overlap + label_overlap)
            false_positives += (len(prediction_words) - overlap + len(prediction_labels) - label_overlap)
            false_negatives += (len(target_words) - overlap + len(target_labels) - label_overlap)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Append individual metrics to lists
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

    # Calculate averages
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1_score = np.mean(f1_score_list)

    return avg_precision, avg_recall, avg_f1_score

# Read data from the files
predictions_file = '/Users/chaitrakaustubh/LegalIE/LegalIE-master/data/SubOrdinationDataset/predictions/T5_SMALL/Prediction_T5_SMALL_bs32.coords'
sample_test_file = '/Users/chaitrakaustubh/LegalIE/LegalIE-master/data/SubOrdinationDataset/gold/test.coords'

sample_test_data, predictions_data = read_file(sample_test_file), read_file(predictions_file)

# Calculate precision, recall, and F1-score and print the averages
avg_precision, avg_recall, avg_f1_score = calculate_precision_recall_f1(sample_test_data[0], predictions_data[1])
print("Precision:", avg_precision)
print("Recall:", avg_recall)
print("F1 Score:", avg_f1_score)