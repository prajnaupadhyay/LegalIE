# -*- coding: utf-8 -*-
'''
Usage: python run.py test train.txt model_dir test.txt predictions.txt T5
Usage: python run.py train train.txt model_dir test.txt predictions.txt T5

'''
# !pip install transformers
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import sys
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.metrics import f1_score
import numpy as np
from transformers import AutoModel, set_seed
import spacy
from scipy.optimize import linear_sum_assignment

nlp = spacy.load("en_core_web_sm")


def get_sentences_from_tree_labels(model='T5', tree_label=None):
    relations = ['SUBORDINATION', 'ELABORATION', 'CONDITION', 'LIST',
                 'TEMPORAL', 'PURPOSE', 'RESULT', 'ATTRIBUTION', 'CLAUSE', 'CONTRAST']
    r2 = ["),‘", "\",\"", "\", \"", "’,'", "’,’", "','", "’ ,’", "’, ‘", "' , '", "' ,'", "', ‘", ") )", "))", "), ", ") ,", "‘ , ‘", "’,‘", "', '", "”,”", "', “", "’, '", '\'', 'SUBORDINATION', 'CO/ELABORATION', 'SUB/ELABORATION', 'CO/CONDITION', 'SUB/CONDITION', 'CO/LIST', 'SUB/LIST', 'CO/TEMPORAL',
          'CO/DISJUNCTION', 'SUB/TEMPORAL', 'CO/PURPOSE', 'SUB/PURPOSE', 'CO/RESULT', 'SUB/RESULT', 'CO/CLAUSE', 'SUB/CLAUSE', 'CO/CONTRAST', 'SUB/CONTRAST', 'SUB/DISJUNCTION', "CO/LSIT", 'SUB/ATTRIBUTION', 'CO/ATTRIBUTION', 'SUB/SPATIAL', 'SUB/BACKGROUND', ")'", "SUB/CAUSE", "SUB / ELABORATION"]
    if tree_label == "NONE":
        return [""]
    count = tree_label.count("COORDINATION")
    count += tree_label.count("CO/")
    count += tree_label.count("SUB/")
    # Removing " )) from the end
    # if count >= 1:
    #     tree_label = tree_label[:-(count + 2)]
    # if(model == "OpenIE"):
    #     sentences = tree_label.split("\" , \"")
    # else:
    #     sentences = tree_label.split("\", \"")
    # sentences = tree_label.split("\",\"")
    # sentences = "####".join([s.split("\", \"")])
    sentences = tree_label
    new_sentenes = []
    if model in ["T5", "OpenIE", "BART"]:
        for d in r2:
            sentences = "####".join(sentences.split(d))
        new_sentenes = sentences.split("####")
    else:
        print("Invalid model name")
        sys.exit(0)
    for i in range(len(new_sentenes)):
        # print(s)
        if new_sentenes[i].startswith("("):
            new_sentenes[i] = new_sentenes[i][1:]
        new_sentenes[i] = new_sentenes[i].strip()
        new_sentenes[i] = new_sentenes[i].strip("”")
        new_sentenes[i] = new_sentenes[i].strip(",,")
        new_sentenes[i] = new_sentenes[i].strip(",")
        new_sentenes[i] = new_sentenes[i].strip("(")
        new_sentenes[i] = new_sentenes[i].strip(")")
        new_sentenes[i] = new_sentenes[i].strip("\'")
        new_sentenes[i] = new_sentenes[i].strip("‘")
        new_sentenes[i] = new_sentenes[i].strip("’")
        new_sentenes[i] = new_sentenes[i].replace(" .", " ")
        new_sentenes[i] = new_sentenes[i].replace(".", "")
        new_sentenes[i] = new_sentenes[i].lower()
        new_sentenes[i] = new_sentenes[i].replace(" '", "'")
        new_sentenes[i] = " ".join(
            [sent.text for sent in nlp(new_sentenes[i])])
    # if new_sentenes[0] == new_sentenes[1]:
    #     new_sentenes = [""]
    return new_sentenes


def matcher_using_f1(ref_set, pred_set):
    if (len(ref_set) == 0 and len(pred_set) == 0):
        return 1, 1
    elif (len(ref_set) == 0 and pred_set[0] == pred_set[-1]):
        return 1, 1
    elif (len(ref_set) == 0 or len(pred_set) == 0):
        return 0, 0

    # print(pred_set,"\n",ref_set)
    intsec = np.zeros((len(pred_set), len(ref_set)), dtype=np.float16)
    precision = np.zeros((len(pred_set), len(ref_set)), dtype=np.float16)
    recall = np.zeros((len(pred_set), len(ref_set)), dtype=np.float16)
    f1_score = np.zeros((len(pred_set), len(ref_set)), dtype=np.float16)
    for i in range(len(pred_set)):
        for j in range(len(ref_set)):
            intsec[i][j] = len(pred_set[i].intersection(ref_set[j]))
            precision[i][j] = float(intsec[i][j]) / len(pred_set[i])
            recall[i][j] = float(intsec[i][j]) / len(ref_set[j])
            if (precision[i][j] == 0 and recall[i][j] == 0):
                f1_score[i][j] = 0
            else:
                f1_score[i][j] = (2*precision[i][j]*recall[i]
                                  [j])/(precision[i][j] + recall[i][j])
    row_ind, col_ind = linear_sum_assignment(f1_score, maximize=True)

    return (precision[row_ind, col_ind].sum())/len(pred_set), (recall[row_ind, col_ind].sum())/len(ref_set)
    # return row_ind.tolist(), col_ind.tolist()


def get_PoS_tags(sentence):
    if sentence.startswith('Input: '):
        sentence.replace('Input: ', '').strip()
    pos = nlp(sentence)
    sentence = " ".join([" ".join([i.text, i.pos_]) for i in pos])
    return sentence

# Define function to process input file


def process_input_file(file_path, dataset="coord"):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    targets = []
    if dataset.lower() == "carb":
        data = [line.strip() for line in lines]
        print(len(data))
        print("No targets given for CARB dataset")
        targets = data
    else:
        for line in lines:
            line = line.strip()
            if line.startswith('Input: '):
                # line = get_PoS_tags(line)
                data.append(line.replace('Input: ', '').strip())
            elif line.startswith('Prediction: '):
                targets.append(line.replace('Prediction: ', '').strip())
        if len(data) == 0:
            data = [line.strip() for line in lines]
        if len(targets) == 0:
            targets = data

        print(len(targets))
        print(len(data))

    return data, targets


# Define batch encoding function
def batch_encode_fn(batch, tokenizer):
    src_texts = [item["source"] for item in batch]
    tgt_texts = [item["target"] for item in batch]
    inputs = tokenizer(src_texts, padding=True,
                       truncation=True, return_tensors="pt")
    targets = tokenizer(tgt_texts, padding=True,
                        truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    targets = {k: v.to(device) for k, v in targets.items()}
    return inputs, targets


def train(train_dataloader, num_epochs, optimizer, model, output_dir, tokenizer):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_custom_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in train_dataloader:
            optimizer.zero_grad()

            inputs, targets = batch
            outputs = model(**inputs, labels=targets["input_ids"])

            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            predicted_token_ids = torch.argmax(logits, dim=-1)
            decoded_predictions = [tokenizer.decode(
                p.tolist(), skip_special_tokens=True) for p in predicted_token_ids]
            decoded_labels = [l for l in targets]

            # input = torch.randn(3, 5, requires_grad=True)
            # custom_loss = BCEWithLogitsLoss()(input, input)
            precision, recall, count = 0, 0, 0

            for dp, dl in zip(decoded_predictions, decoded_labels):
                predictions_list1 = get_sentences_from_tree_labels(
                    tree_label=" ".join(dp))
                predictions_list = [set((s.strip()).split())
                                    for s in predictions_list1 if s.strip() != ""]
                labels_list1 = get_sentences_from_tree_labels(
                    tree_label=" ".join(dl))
                labels_list = [set((s.strip()).split())
                               for s in labels_list1 if s.strip() != ""]

                # pred_ind, ref_ind = matcher_using_f1(
                #     predictions_list, labels_list)

                # if pred_ind == 1 and ref_ind == 1:
                #     continue
                # if pred_ind == 0 and ref_ind == 0:
                #     continue
                # for i, j in zip(pred_ind, ref_ind):
                #     ep = torch.tensor(tokenizer(
                #         predictions_list1[i], padding=True, truncation=True, return_tensors="pt")["input_ids"], dtype=torch.float)
                #     el = torch.tensor(tokenizer(
                #         labels_list1[i], padding=True, truncation=True, return_tensors="pt")["input_ids"])
                #     custom_loss += BCEWithLogitsLoss()(ep, el)
                p, r = matcher_using_f1(predictions_list, labels_list)
                precision += 1 - p
                recall += 1 - r
                count += 1

            precision, recall = precision/count, recall/count
            loss = (1 + precision + recall) * loss
            # total_custom_loss += custom_loss.item()

            # Flatten the targets and predictions tensors
            flat_targets = targets["input_ids"].flatten()
            flat_predictions = predictions.flatten()

            # Move tensor from CUDA device to CPU
            flat_predictions_cpu = flat_predictions.cpu()
            flat_targets_cpu = flat_targets.cpu()

            flat_predictions_np = flat_predictions_cpu.numpy()  # Convert tensor to NumPy array
            flat_targets_np = flat_targets_cpu.numpy()

            correct = (flat_predictions_np == flat_targets_np).sum().item()
            total_correct += correct
            total_samples += flat_targets.size(0)

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_dataloader)
        avg_custom_loss = total_custom_loss / len(train_dataloader)
        accuracy = total_correct / total_samples
        accuracy_lib = accuracy_score(flat_targets_np, flat_predictions_np)
        f1 = f1_score(flat_targets_np, flat_predictions_np, average='micro')
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss}, Accuracy: {accuracy}, Accuracy_lib: {accuracy_lib}, F1_score: {f1}")

    # Save the trained model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# Define the function to write predictions to a file
def write_predictions_to_file(file_path, inputs, predictions):
    with open(file_path, 'w', encoding='utf-8') as file:
        for i in range(len(inputs)):
            file.write(
                "Input: " + " ".join([sent.text for sent in nlp(inputs[i])]) + "\n")
            # predictions[i] = predictions[i].replace("..", ".")
            # predictions[i] = predictions[i].replace(",.", ".")
            predictions[i] = " ".join(
                [sent.text for sent in nlp(predictions[i])])
            predictions[i] = predictions[i].replace(
                "COORDINATION ( \"", "COORDINATION(\"")
            predictions[i] = predictions[i].replace(
                "COORDINATIONAL ( \"", "COORDINATION(\"")
            predictions[i] = predictions[i].replace(". \"", ".\"")
            # predictions[i] = predictions[i].replace(" / ", "\\/")
            predictions[i] = predictions[i].replace(" - ", "-")
            predictions[i] = predictions[i].replace(") )", "))")

            file.write("Prediction: " + predictions[i] + "\n")
            # print("Prediction:", predictions[i])
            file.write("\n")


# Evaluation loop
def test(test_dataloader, model, output_file_path, tokenizer):
    model.eval()
    input_texts = []
    predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs, batch_targets = batch
            inputs = {k: v.to(device)
                      for k, v in inputs.items()}  # Move inputs to device

            outputs = model.generate(input_ids=inputs["input_ids"].to(
                device), max_length=1000)  # Generate predictions

            # Decode the generated output and convert to text
            batch_predictions = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                                 for output in outputs]

            # Append batch inputs and predictions to the overall lists
            input_texts.extend([tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                                for input_ids in inputs["input_ids"]])
            predictions.extend(batch_predictions)

            # Print the current batch predictions
            print("Batch Predictions:", batch_predictions)

    # Write inputs and predictions to file
    write_predictions_to_file(output_file_path, input_texts, predictions)


def prepare_train(model_name, bs=3):
    # Load and preprocess your training data (from input file)
    input_file_path = sys.argv[2]
    data, targets = process_input_file(input_file_path)

    model = None
    if model_name.upper() == 'BART':
        model = BartForConditionalGeneration.from_pretrained(
            "lucadiliello/bart-small").to(device)
    elif model_name.upper() == 'BERT':
        model = AutoModel.from_pretrained(
            "nlpaueb/legal-bert-base-uncased").to(device)
    elif model_name.upper() == 'T5':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-small").to(device)
    else:
        print('Please enter a valid model name')

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    print('optimizer done')
    # Define your training dataset
    train_dataset = [{"source": data[i], "target": targets[i]}
                     for i in range(len(data))]

    # Define your training dataloader
    batch_size = bs
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=lambda batch: batch_encode_fn(batch, tokenizer))
    num_epochs = 30

    # define directory to store the model
    output_dir = sys.argv[3]

    train(train_dataloader, num_epochs, optimizer, model, output_dir, tokenizer)


def prepare_test(model_name, carb=False, bs=3):
    # load saved model and tokenizer
    model = None
    if model_name.upper() == 'BART':
        model = BartForConditionalGeneration.from_pretrained(
            sys.argv[3]).to(device)
    elif model_name.upper() == 'T5':
        model = AutoModelForSeq2SeqLM.from_pretrained(sys.argv[3]).to(device)

    # Load and preprocess your test data (from input file)

    test_file_path = sys.argv[4]
    test_data, test_targets = process_input_file(test_file_path, carb)

    # Define your test dataset
    test_dataset = [{"source": test_data[i], "target": test_targets[i]}
                    for i in range(len(test_data))]

    batch_size = bs

    # Define your test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=lambda batch: batch_encode_fn(batch, tokenizer))

    # read the output file path
    output_file_path = sys.argv[5]

    # test function to test the model and write predictions to file
    test(test_dataloader, model, output_file_path, tokenizer)


if __name__ == '__main__':
    # Set up device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(sys.argv) < 8:
        print('Usage: python run.py test train.txt model_dir test.txt predictions.txt BART or T5 batch_size seed')
        sys.exit(1)
    set_seed(int(sys.argv[8]))
    batch_size = int(sys.argv[7])
    # Choose model
    tokenizer = None
    if sys.argv[6].upper() == 'BART':
        tokenizer = BartTokenizer.from_pretrained("lucadiliello/bart-small")
    elif sys.argv[6].upper() == 'BERT':
        tokenizer = AutoTokenizer.from_pretrained(
            "nlpaueb/legal-bert-base-uncased")
    elif sys.argv[6].upper() == 'T5':
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    else:
        print('Wrong model name. Use BART ot T5')
        sys.exit(1)

    # Choose Dataset
    dataset = "coord"
    # if len(sys.argv) == 8 and sys.argv[7].lower() in ["coord", "subord", "carb"]:
    #     dataset = sys.argv[7].lower()
    #     print("Dataset: ", dataset.upper())
    # else:
    #     print("Wrong Dataset")
    #     sys.exit(1)

    # Choose task
    if sys.argv[1] == 'train-test':
        prepare_train(sys.argv[6], batch_size)
        prepare_test(sys.argv[6], dataset, batch_size)
    elif sys.argv[1] == 'train':
        prepare_train(sys.argv[6], batch_size)
    elif sys.argv[1] == 'test':
        prepare_test(sys.argv[6], dataset, batch_size)
    elif sys.argv[1] == 'predict':
        print('predict')
