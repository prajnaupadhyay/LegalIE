from preprocess import get_sentences_from_tree_labels
import numpy as np
import sys
from scipy.optimize import linear_sum_assignment
import re
import spacy

class Preprocessor:
    
    @classmethod
    def get_copy_file(cls):
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
        
    @classmethod
    def get_mod2_file(cls):
        # path1 = "/media/sankalp/DATA/Legal_NLP/LegalIE/data/CoordinationDataSet/gold/test.coord"
        # path2 = "/media/sankalp/DATA/Legal_NLP/LegalIE/data/CoordinationDataSet/gold/test_mod2.coord"

        path1 = "/media/sankalp/DATA/Legal_NLP/LegalIE/data/CoordinationDataSet/input/train.coord"
        path2 = "/media/sankalp/DATA/Legal_NLP/LegalIE/data/CoordinationDataSet/input/train_mod2.coord"


        fr = open(path1, "r")
        fw = open(path2, "w")
        f = fr.readlines()

        for i in range(len(f)):
            if i%3 == 0:
                f[i] = "Input: " + f[i][1:]
                # print(f[i])
            if i%3 == 1:
                f[i] = "Prediction: " + f[i]
            f[i] = f[i].replace(" ' ", "' ")
            f[i] = f[i].replace(" .", ".")
            f[i] = f[i].replace(" , ", ", ")
            f[i] = f[i].replace(" ?", "?")
            f[i] = f[i].replace("\\/", "/")
            
            f[i] = f[i].replace(" 's", "'s")
            f[i] = f[i].replace(" 'd", "'d")
            f[i] = f[i].replace(" 'm", "'m")
            f[i] = f[i].replace(" 're", "'re")
            f[i] = f[i].replace(" 've", "'ve")
            f[i] = f[i].replace(" 'll", "'ll")
            
            f[i] = f[i].replace(" n't", "n't")
            # f[i] = f[i].replace("wo n't", "won't")
            # f[i] = f[i].replace("do n't", "don't")
            # f[i] = f[i].replace("ca n't", "can't")
            # f[i] = f[i].replace("is n't", "isn't")
            # f[i] = f[i].replace("did n't", "didn't")
            # f[i] = f[i].replace("are n't", "aren't")
            # f[i] = f[i].replace("was n't", "wasn't")
            # f[i] = f[i].replace("had n't", "hadn't")
            # f[i] = f[i].replace("have n't", "haven't")
            # f[i] = f[i].replace("were n't", "weren't")
            # f[i] = f[i].replace("does n't", "doesn't")
            # f[i] = f[i].replace("would n't", "wouldn't")
            # f[i] = f[i].replace("could n't", "couldn't")
            # f[i] = f[i].replace("should n't", "shouldn't")
            fw.write(f[i])
        fr.close()
        fw.close()

    @classmethod
    def postprocess_on_mod_file(cls, s):
        s.replace(" 's", "'s")
        s.replace("\\/", "/")
        s.replace("&", " & ")
        s.replace(" n't", "n't")
        s.replace(" ' ", "'")

    @classmethod
    def preprocess_SubordData(cls):
        # path1 = "data/SubordinationDataSet/input/train.txt"
        # path2 = "data/SubordinationDataSet/input/train_IP.txt"
        
        path1 = "data/SubordinationDataSet/gold/test.txt"
        path2 = "data/SubordinationDataSet/gold/test_IP.txt"

        fr = open(path1, "r")
        fw = open(path2, "w")
        f = fr.readlines()

        for i in range(len(f)):
            if i%2 == 0:
                f[i] = "Input: " + f[i][1:]
                # print(f[i])
            if i%2 == 1:
                f[i] = "Prediction: " + f[i] + "\n"
            fw.write(f[i])
        fr.close()
        fw.close() 
        
    @classmethod
    def get_in_openie_format(cls, path1, path2):
        # path1 = "data/CoordinationDataSet/output/predictions/Prediction_T5_base.coord"
        # path2 = "data/CoordinationDataSet/output/predictions/Prediction_T5_base.coord"

        fr = open(path1, "r")
        fw = open(path2, "w")
        predictions = fr.readlines()
        nlp = spacy.load('en_core_web_sm')

        for i in range(len(predictions)):
            # predictions[i] = re.sub(r'([a-zA-Z])\.', r'\1 .', predictions[i])
            # predictions[i] = re.sub(r'\\([0-9])\.', r'\1 .', predictions[i])
            # predictions[i] = predictions[i].replace("Inc .", "Inc.")
            # predictions[i] = predictions[i].replace("Co .", "Co.")
            # predictions[i] = predictions[i].replace("Mr .", "Mr.")
            # predictions[i] = predictions[i].replace("Dr .", "Dr.")            
            # predictions[i] = predictions[i].replace(", ", " , ")
            # predictions[i] = predictions[i].replace("/", "\\/")
            # predictions[i] = predictions[i].replace("'s", " 's")
            # predictions[i] = predictions[i].replace("n't", " n't")
            doc = nlp(predictions[i])
            fw.write(" ".join([sent.text for sent in doc]) + "\n")
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
                # print([s.strip() for s in new_sentences if s.strip() != ""])
                sentences = [set((s.strip()).split()) for s in new_sentences if s.strip() != ""]
                # print(sentences)
                ref.append(sentences)
                
        if model == "OpenIE":
            pred = pred_file.read().split('\n\n')
            for i in range(len(pred)):
                # print(pred[i])
                _t = [s.strip() for s in pred[i].splitlines() if s.strip() != ""]
                # _t = _t[1:]
                # print(_t)
                pred[i] = [set((s.strip()).split()) for s in sorted(_t[1:])]
                # print(pred[i])
            pred = pred[:-1]
        else:        
            for line in pred_file.readlines():
                if line.startswith("Prediction: "):
                    prediction = line.replace("Prediction: ", "")[:-1]
                    new_sentences = sorted(get_sentences_from_tree_labels(model, prediction))
                    sentences = [set((s.strip()).split()) for s in new_sentences  if s.strip() != ""]
                    # print(sentences)
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
        idx, intsec, precision, recall = [], [], [], []
        if (len(ref_set) < len(pred_set)):
            idx = np.argmax(mat, axis=1)
            if len(np.unique(idx)) != len(ref_set):
                raise Exception("NN without Replacement failed")
            intsec = [len(pred_set[i].intersection(ref_set[idx[i]])) for i in range(len(idx))]
            precision = [float(intsec[i]/ len(pred_set[i])) for i in range(len(intsec))]
            recall = [float(intsec[i]/ len(ref_set[idx[i]])) for i in range(len(intsec))]
        else:
            idx = np.argmax(mat, axis=0)
            if len(np.unique(idx)) != len(pred_set):
                raise Exception("NN without Replacement failed")
            intsec = [len(ref_set[i].intersection(pred_set[idx[i]])) for i in range(len(idx))]
            precision = [float(intsec[i]/ len(pred_set[idx[i]])) for i in range(len(intsec))]
            recall = [float(intsec[i]/ len(ref_set[i])) for i in range(len(intsec))]

        print(intsec, precision, recall, len(pred_set), len(ref_set))
        return sum(precision)/len(pred_set), sum(recall)/len(ref_set)
    
    @classmethod
    def matcher2(cls, ref_set, pred_set):
        if(len(ref_set) == 0 and len(pred_set) == 0):
            return 1.0, 1.0
        elif(len(ref_set) == 0 or len(pred_set) == 0):
            return 0.0, 0.0
        mat = np.zeros((len(pred_set), len(ref_set)), dtype=np.float16)
        for i in range(len(pred_set)):
            for j in range(len(ref_set)):
                mat[i][j] = float(len(pred_set[i].intersection(ref_set[j]))) / len(pred_set[i].union(ref_set[j]))
        # row_ind, col_ind = None, None
        row_ind, col_ind = linear_sum_assignment(mat, maximize = True)
        # print(row_ind, col_ind)
        intsec = [len(pred_set[i].intersection(ref_set[j])) for i,j in zip(row_ind, col_ind)]
        precision = [float(intsec[i])/len(pred_set[i]) for i in range(len(intsec))]
        
        row_ind, col_ind = linear_sum_assignment(mat.T, maximize = True)
        # print(row_ind, col_ind)
        intsec = [len(ref_set[i].intersection(pred_set[j])) for i,j in zip(row_ind, col_ind)]
        recall = [float(intsec[i])/len(ref_set[i]) for i in range(len(intsec))]
        
        return sum(precision)/len(pred_set), sum(recall)/len(ref_set)
    
    @classmethod
    def matcher3(cls, ref_set, pred_set):
        if(len(ref_set) == 0 and len(pred_set) == 0):
            return 1.0, 1.0
        elif(len(ref_set) == 0 or len(pred_set) == 0):
            return 0.0, 0.0
        intsec = np.zeros((len(pred_set), len(ref_set)), dtype=np.float16)
        precision = np.zeros((len(pred_set), len(ref_set)), dtype=np.float16)
        recall = np.zeros((len(pred_set), len(ref_set)), dtype=np.float16)
        f1_score = np.zeros((len(pred_set), len(ref_set)), dtype=np.float16)
        for i in range(len(pred_set)):
            for j in range(len(ref_set)):
                intsec[i][j] = len(pred_set[i].intersection(ref_set[j]))
                precision[i][j] = float(intsec[i][j]) / len(pred_set[i])
                recall[i][j] = float(intsec[i][j]) / len(ref_set[j])
                if(precision[i][j] ==0 and recall[i][j]==0 ):
                    f1_score[i][j] = 0
                else:
                    f1_score[i][j] = (2*precision[i][j]*recall[i][j])/(precision[i][j] + recall[i][j])
                
        row_ind, col_ind = linear_sum_assignment(f1_score, maximize = True)
        
        return (precision[row_ind, col_ind].sum())/len(pred_set), (recall[row_ind, col_ind].sum())/len(ref_set)
                
    @classmethod
    def scorer(cls, model, ref_path, pred_path):
        ref, pred = cls.reader(model, ref_path, pred_path)
        precision, recall, f1_score, f1 = 0, 0, 0, 0
        for i in range(len(ref)):
            p, r = cls.matcher3(ref[i], pred[i])
            # print(p, r)
            if(p != 0 and r != 0):
                f1 = 2*p*r/(p+r)
            precision += p
            recall += r
            f1_score += f1
        return precision/len(ref), recall/len(ref), f1_score/len(ref)

if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("Usage: python3 wire.py T5/BART/OpenIE <refernce_file> <prediction_file>")
    #     exit(0)
    # precision, recall, f1_score =  wire57_scorer.scorer(sys.argv[1], sys.argv[2], sys.argv[3])
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("F1 Score: ", f1_score)
    
    # Preprocessor.get_mod2_file()
    # Preprocessor.preprocess_SubordData()
    
    if (len(sys.argv) != 3):
        print("Usage: python3 wire57.py <input_file> <output_file>")
        exit(0)
    Preprocessor.get_in_openie_format(sys.argv[1], sys.argv[2])