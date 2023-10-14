import numpy as np
import sys
from wire57 import wire57_scorer

def get_sentences_from_tree_labels(model = 'T5', tree_label = None):
    relations = ['SUBORDINATION', 'ELABORATION', 'CONDITION', 'LIST', 'TEMPORAL', 'PURPOSE', 'RESULT', 'ATTRIBUTION', 'CLAUSE', 'CONTRAST']
    r2 = ["),‘", "\",\"", "\", \"", "’,'", "’,’", "','", "’ ,’", "’, ‘", "' , '", "' ,'", "', ‘", ") )", "))" , "), ", ") ," ,"‘ , ‘", "’,‘", "', '" , "”,”" , "', “" , "’, '" , 'SUBORDINATION','CO/ELABORATION', 'SUB/ELABORATION', 'CO/CONDITION', 'SUB/CONDITION', 'CO/LIST', 'SUB/LIST', 'CO/TEMPORAL', 'CO/DISJUNCTION', 'SUB/TEMPORAL', 'CO/PURPOSE', 'SUB/PURPOSE', 'CO/RESULT', 'SUB/RESULT', 'CO/CLAUSE', 'SUB/CLAUSE', 'CO/CONTRAST', 'SUB/CONTRAST', 'SUB/DISJUNCTION', "CO/LSIT", 'SUB/ATTRIBUTION', 'CO/ATTRIBUTION', 'SUB/SPATIAL', 'SUB/BACKGROUND', ")'", "SUB/CAUSE", "SUB / ELABORATION"]
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
    # if new_sentenes[0] == new_sentenes[1]:
    #     new_sentenes = [""]   
    return new_sentenes

class overlap_scorer(wire57_scorer):
    @classmethod
    def reader(cls, model, ref_path, pred_path):
        ref_file = open(ref_path, "r")
        pred_file = open(pred_path, "r")
        ref = []
        pred = []
        for line in ref_file:
            if line.startswith("Prediction: "):
                prediction = line.replace("Prediction: ", "")[:-1]
                new_sentences = get_sentences_from_tree_labels(model, prediction)
                # print(new_sentences)
                # print([s.strip() for s in new_sentences if s.strip() not in ["", "."]])
                sentences = [set((s.strip()).split()) for s in new_sentences if s.strip() not in ["", "."]]
                # print(sentences)
                ref.append(sentences)
        for line in pred_file.readlines():
            if line.startswith("Prediction: "):
                prediction = line.replace("Prediction: ", "")[:-1]
                new_sentences = get_sentences_from_tree_labels(model, prediction)
                # print(new_sentences)
                sentences = [set((s.strip()).split()) for s in new_sentences  if s.strip() not in ["", "."]]
                # print(sentences)
                pred.append(sentences)
        if len(ref) != len(pred):
            raise Exception(f"Number of sentences in reference {len(ref)} and prediction {len(pred)} are not equal")
        return ref, pred

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 wire.py T5/BART/OpenIE <refernce_file> <prediction_file>")
        exit(0)
    # ref_file = open(sys.argv[3], "r")
    # ref = []
    # for line in ref_file:
    #     if line.startswith("Prediction: "):
    #             prediction = line.replace("Prediction: ", "")[:-1]
    #             new_sentences = get_sentences_from_tree_labels(sys.argv[1], prediction)
    #             sentences = [s.strip() for s in new_sentences if s.strip() not in ["", "("]]
    #             # for s in sentences:
    #             #     print(s)
    #             # print("\n")
    #             ref.append(sentences)
    # for r in ref:
    #     for s in r:
    #         print(s)
    #     print("\n")
    
    precision, recall, f1_score =  overlap_scorer.scorer(sys.argv[1], sys.argv[2], sys.argv[3])
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1_score)