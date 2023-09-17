from preprocess import get_sentences_from_tree_labels
import numpy as np
import sys

def openIE_to_generative():
    path1 = "/media/sankalp/DATA/Legal_NLP/LegalIE/data/CoordinationDataSet/gold/test.coord"
    path2 = "/media/sankalp/DATA/Legal_NLP/LegalIE/data/CoordinationDataSet/gold/test_copy.coord"

    fr = open(path1, "r")
    fw = open(path2, "w")
    f = fr.readlines()

    for i in range(len(f)):
        if i%3 == 0:
            f[i] = "Input: " + f[i][1:]
            # print(f[i])
        if i%3 == 1:
            f[i] = "Prediction: " + f[i]
        fw.write(f[i])
    fr.close()
    fw.close() 
    
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
                new_sentences = sorted(get_sentences_from_tree_labels(model, prediction))
                sentences = [set((s.strip()).split()) for s in new_sentences  if s.strip() != ""]
                # print(sentences)
                ref.append(sentences)
        for line in pred_file.readlines():
            if line.startswith("Prediction: "):
                prediction = line.replace("Prediction: ", "")[:-1]
                new_sentences = sorted(get_sentences_from_tree_labels(model, prediction))
                sentences = [set((s.strip()).split()) for s in new_sentences  if s.strip() != ""]
                pred.append(sentences)
        if len(ref) != len(pred):
            raise Exception(f"Number of sentences in reference {len(ref)} and prediction {len(pred)} are not equal")
        return ref, pred
    
    @classmethod
    def matcher(cls, ref_set, pred_set):
        # print("Length of ref_set: ", len(ref_set), "Length of pred_set: ", len(pred_set))
        if(len(ref_set) == 0 and len(pred_set) == 0):
            return 1.0, 1.0
        elif(len(ref_set) == 0 or len(pred_set) == 0):
            return 0.0, 0.0
        mat = np.zeros((len(pred_set), len(ref_set)), dtype=np.float16)
        for i in range(len(pred_set)):
            for j in range(len(ref_set)):
                mat[i][j] = float(len(pred_set[i].intersection(ref_set[j]))) / len(pred_set[i].union(ref_set[j]))
                # print(mat[i][j])
        idx = np.argmax(mat, axis=1)
        intsec = [len(pred_set[i].intersection(ref_set[idx[i]])) for i in range(len(idx))]
        precision = [float(intsec[i]/ len(pred_set[i])) for i in range(len(intsec))]
        
        idx = np.argmax(mat, axis=0)
        intsec = [len(ref_set[i].intersection(pred_set[idx[i]])) for i in range(len(idx))]
        recall = [float(intsec[i]/ len(ref_set[i])) for i in range(len(intsec))]
        # print(intsec, recall, len(ref_set))
        return sum(precision)/len(pred_set), sum(recall)/len(ref_set)
           
    @classmethod
    def scorer(cls, model, ref_path, pred_path):
        ref, pred = cls.reader(model, ref_path, pred_path)
        precision, recall, f1_score, f1 = 0, 0, 0, 0
        for i in range(len(ref)):
            p, r = cls.matcher(ref[i], pred[i])
            # print(p, r)
            if(p != 0 and r != 0):
                f1 = 2*p*r/(p+r)
            precision += p
            recall += r
            f1_score += f1
        return precision/len(ref), recall/len(ref), f1_score/len(ref)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 wire.py T5 <refernce_file> <prediction_file>")
        exit(0)
    precision, recall, f1_score =  wire57_scorer.scorer(sys.argv[1], sys.argv[2], sys.argv[3])
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1_score)