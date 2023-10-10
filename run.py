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
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.metrics import f1_score
import numpy as np
from transformers import AutoModel
import spacy

def get_PoS_tags(sentence):
    if sentence.startswith('Input: '):
        sentence.replace('Input: ', '').strip()
    nlp = spacy.load("en_core_web_sm")
    pos = nlp(sentence)
    sentence += (" " + " ".join([i.pos_ for i in pos]))
    return sentence

# Define function to process input file
def process_input_file(file_path, dataset = "coord"):
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
    inputs = tokenizer(src_texts, padding=True, truncation=True, return_tensors="pt")
    targets = tokenizer(tgt_texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    targets = {k: v.to(device) for k, v in targets.items()}
    return inputs, targets


def train(train_dataloader, num_epochs, optimizer, model, output_dir, tokenizer):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
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
    nlp = spacy.load("en_core_web_sm")
    with open(file_path, 'w', encoding='utf-8') as file:
        for i in range(len(inputs)):
            file.write("Input: " + " ".join([sent.text for sent in nlp(inputs[i])]) + "\n")
            # predictions[i] = predictions[i].replace("..", ".")
            # predictions[i] = predictions[i].replace(",.", ".")
            predictions[i] =  " ".join([sent.text for sent in nlp(predictions[i])]) 
            predictions[i] = predictions[i].replace("COORDINATION ( \"", "COORDINATION(\"")
            predictions[i] = predictions[i].replace("COORDINATIONAL ( \"", "COORDINATION(\"")
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
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device

            outputs = model.generate(input_ids=inputs["input_ids"].to(device), max_length=1000)  # Generate predictions

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


def prepare_train(model_name):
    # Load and preprocess your training data (from input file)
    input_file_path = sys.argv[2]
    data, targets = process_input_file(input_file_path)
    
    model = None
    if model_name.upper() == 'BART':
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
    elif model_name.upper() == 'T5':
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)
    else:
        print('Please enter a valid model name')

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    print('optimizer done')
    # Define your training dataset
    train_dataset = [{"source": data[i], "target": targets[i]} for i in range(len(data))]

    # Define your training dataloader
    batch_size = 3
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn= lambda batch: batch_encode_fn(batch, tokenizer))
    num_epochs = 30

    # define directory to store the model
    output_dir = sys.argv[3]

    train(train_dataloader, num_epochs, optimizer, model, output_dir, tokenizer)


def prepare_test(model_name, carb = False):
    # load saved model and tokenizer
    model = None
    if model_name.upper() == 'BART':
        model = BartForConditionalGeneration.from_pretrained(sys.argv[3]).to(device)
    elif model_name.upper() == 'T5':
        model = AutoModelForSeq2SeqLM.from_pretrained(sys.argv[3]).to(device)

    # Load and preprocess your test data (from input file)

    test_file_path = sys.argv[4]
    test_data, test_targets = process_input_file(test_file_path, carb)

    # Define your test dataset
    test_dataset = [{"source": test_data[i], "target": test_targets[i]} for i in range(len(test_data))]

    batch_size = 3

    # Define your test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn= lambda batch: batch_encode_fn(batch, tokenizer))

    # read the output file path
    output_file_path = sys.argv[5]

    # test function to test the model and write predictions to file
    test(test_dataloader, model, output_file_path, tokenizer)


if __name__ == '__main__':
    # Set up device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(sys.argv) < 7:
        print('Usage: python run.py test train.txt model_dir test.txt predictions.txt BART or T5 [CARB]')
        sys.exit(1)
    
    # Choose model    
    tokenizer = None
    if sys.argv[6].upper() == 'BART':
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
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
        prepare_train(sys.argv[6])
        prepare_test(sys.argv[6], dataset)
    elif sys.argv[1] == 'train':
        prepare_train(sys.argv[6])
    elif sys.argv[1] == 'test':
        prepare_test(sys.argv[6], dataset)
    elif sys.argv[1] == 'predict':
        print('predict')
