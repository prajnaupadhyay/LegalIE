# -*- coding: utf-8 -*-
'''
Usage: python run.py test train.txt model_dir test.txt predictions.txt T5
Usage: python run.py train train.txt model_dir test.txt predictions.txt T5

'''
# !pip install transformers
import torch
from torch.optim import AdamW
import sys
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, get_constant_schedule
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.metrics import f1_score
import numpy as np
from transformers import AutoModel, set_seed
import spacy
from Utils.wire57.wire57scorer import matcher_using_f1
from Utils.overlap_score import get_sentences_from_tree_labels


class SubordDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class LeGenTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = outputs.get("loss")
        print(loss)

        # decode predictions and labels
        predicted_token_ids = torch.argmax(logits, dim=-1)
        decoded_predictions = [tokenizer.decode(
            p.tolist(), skip_special_tokens=True) for p in predicted_token_ids]
        decoded_labels = [tokenizer.decode(
            l.tolist(), skip_special_tokens=True) for l in labels]

        custom_loss = 0.0

        for dp, dl in zip(decoded_predictions, decoded_labels):
            predictions_list = get_sentences_from_tree_labels(
                tree_label=" ".join(dp))
            labels_list = get_sentences_from_tree_labels(
                tree_label=" ".join(dl))

            p, r = matcher_using_f1(predictions_list, labels_list)
            custom_loss += ((1 - p) + (1 - r))

        return (custom_loss, outputs) if return_outputs else loss


def get_PoS_tags(sentence):
    if sentence.startswith('Input: '):
        sentence.replace('Input: ', '').strip()
    nlp = spacy.load("en_core_web_sm")
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


# Define the function to write predictions to a file
def write_predictions_to_file(file_path, inputs, predictions):
    nlp = spacy.load("en_core_web_sm")
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
        model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
    elif model_name.upper() == 'T5':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-small").to(device)
    else:
        print('Please enter a valid model name')

    train_encodings = tokenizer(data, padding=True, truncation=True)
    train_dataset = SubordDataset(train_encodings, targets)

    training_args = TrainingArguments(
        output_dir=sys.argv[3],          # output directory
        num_train_epochs=30,              # total number of training epochs
        per_device_train_batch_size=bs,  # batch size per device during training
        logging_dir='./trainer_logs',            # directory for storing logs
        logging_steps=5,
    )

    optimizer = AdamW(model.parameters(), lr=1e-5)
    schedular = get_constant_schedule(optimizer)

    trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        tokenizer=tokenizer,
        optimizers=(optimizer, schedular)
    )

    trainer.train()


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
