# import argparse

# parser = argparse.ArgumentParser(description="Run LeGen model")
# parser.add_argument("action",
#                     help= "Choose action for the model",
#                     choices= ['train', 'test', 'train-test'])
# parser.add_argument("--train_file",
#                     help = "Path to traning dataset file. Generally it will be in data/<dataset>/input/")
# parser.add_argument("--model_dir",
#                     help= "Path to model directory. If action is train then model will stored here. If it is test then model saved here will be used to make predictions")
# parser.add_argument("--test_file",
#                     help= "Path to test dataset file. Generally it will be in data/<dataset>/gold/")
# parser.add_argument("")

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import spacy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

nlp = spacy.load("en_core_web_sm")


def get_discourse_tree(text):
    sentences = " ".join([t.text for t in nlp(text)])

    input_ids = tokenizer.encode(
        sentences, max_length=384, truncation=True, return_tensors='pt')
    outputs = model(**input_ids, )

    loss = outputs.loss
    total_loss += loss.item()

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # Flatten the targets and predictions tensors
    flat_predictions = predictions.flatten()
    # outputs = model.generate(input_ids=input_ids, max_length=128)

    answer = [tokenizer.decode(output, skip_special_tokens=True)
              for output in predictions]
    return " ".join(answer)

text = """ If balance amount in the account of a deceased is higher than ₹ 150,000 then the nominee or legal heir has to prove the identity to claim the amount . """

print(get_discourse_tree(text))