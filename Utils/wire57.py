from preprocess import get_sentences_from_tree_labels
import numpy as np
import sys

class wire57_scorer:
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
                sentences = [set(s.strip()) for s in new_sentences  if s.strip() != ""]
                ref.append(sentences)
        for line in pred_file:
            if line.startswith("Prediction: "):
                prediction = line.replace("Prediction: ", "")[:-1]
                new_sentences = get_sentences_from_tree_labels(model, prediction)
                sentences = [set(s.strip()) for s in new_sentences  if s.strip() != ""]
                pred.append(sentences)
        if len(ref) != len(pred):
            raise Exception("Number of sentences in reference and prediction are not equal")
        return ref, pred
    
    @classmethod
    def matcher(cls, ref_set, pred_set):
        mat = np.zeros((len(pred_set), len(ref_set)), dtype=np.float16)
        for i in range(len(pred_set)):
            for j in range(len(ref_set)):
                mat[i][j] = float(len(pred_set[i].intersection(ref_set[j]))) / len(pred_set[i].union(ref_set[j]))
        idx = np.argmax(mat, axis=1)
        intsec = [len(pred_set[i].intersection(ref_set[idx[i]])) for i in len(idx)]
        precision = sum([float(i/ len(pred_set[i])) for i in intsec])
        recall = sum([float(i/ len(ref_set[i])) for i in intsec])
        return precision/len(pred_set), recall/len(ref_set)
           
    @classmethod
    def scorer(cls, model, ref_path, pred_path):
        ref, pred = cls.reader(model, ref_path, pred_path)
        precision, recall, f1_score = 0, 0, 0
        for i in range(len(ref)):
            p, r = cls.matcher(ref[i], pred[i])
            f1 = 2*p*r/(p+r)
            precision += p
            recall += r
            f1_score += f1
        return precision/len(ref), recall/len(ref), f1_score/len(ref)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 wire.py T5 <refernce_file> <prediction_file>")
        exit(0)
    print(wire57_scorer.scorer(sys.argv[1], sys.argv[2], sys.argv[3]))