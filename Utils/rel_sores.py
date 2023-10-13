from overlap_score import get_sentences_from_tree_labels, overlap_scorer
import numpy as np
import pandas as pd
import sys

class rel_overlap_scorer(overlap_scorer):
    @classmethod
    def reader(cls, model, ref_path, pred_path):
        ref_file = open(ref_path, "r").read()
        pred_file = open(pred_path, "r")
        rel = ['CO/ELABORATION', 'SUB/ELABORATION', 'CO/CONDITION', 'SUB/CONDITION', 'CO/LIST', 'SUB/LIST', 'CO/TEMPORAL', 'CO/DISJUNCTION', 'SUB/TEMPORAL', 'CO/PURPOSE', 'SUB/PURPOSE', 'CO/RESULT', 'SUB/RESULT', 'CO/CLAUSE', 'SUB/CLAUSE', 'CO/CONTRAST', 'SUB/CONTRAST', 'SUB/DISJUNCTION', "CO/LSIT", 'SUB/ATTRIBUTION', 'CO/ATTRIBUTION', 'SUB/SPATIAL', 'SUB/BACKGROUND', 'SUB/CAUSE']
        ref = []
        pred = []
        count = np.zeros((ref_file.count("Prediction: "), len(rel)), np.int8)
        c = 0
        ref_file = open(ref_path, "r")
        for j, line in enumerate(ref_file):
            if line.startswith("Prediction: "):
                prediction = line.replace("Prediction: ", "")[:-1]
                count[c] = np.array([1 if i in prediction else 0 for i in rel ])
                c += 1
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
        # refs = [[] for i in range(max(count)+1)]
        # preds = [[] for i in range(max(count)+1)]
        # for i in range(len(count)):
        #     refs[count[i]].append(ref[i])
        #     preds[count[i]].append(pred[i])
        return ref, pred, count
    
    @classmethod
    def scorer(cls, model, ref_path, pred_path):
        ref, pred, count = cls.reader(model, ref_path, pred_path)
        f1 =  0
        precision = np.zeros(count.shape[1])
        recall = np.zeros(count.shape[1])
        f1_score = np.zeros(count.shape[1])
        for i in range(len(ref)):
            p, r = cls.matcher_using_f1(ref[i], pred[i])
            # print(p, r)
            if(p != 0 and r != 0):
                f1 = 2*p*r/(p+r)
            else:
                f1 = 0
            for j in range(count.shape[1]):
                precision[j] += p*count[i][j]
                recall[j] += r*count[i][j]
                f1_score[j] += f1*count[i][j]
        tot = np.sum(count, axis=0)
        precision = np.array(precision)
        recall = np.array(recall)
        f1_score = np.array(f1_score)
        return np.divide(precision, tot, where=tot!=0), np.divide(recall, tot, where=tot!=0), np.divide(f1_score, tot, where=tot!=0)
    
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 wire.py T5/BART/OpenIE <refernce_file> <prediction_file> <result_csv>")
        exit(0)
    precision, recall, f1_score =  rel_overlap_scorer.scorer(sys.argv[1], sys.argv[2], sys.argv[3])
    rel = ['CO/ELABORATION', 'SUB/ELABORATION', 'CO/CONDITION', 'SUB/CONDITION', 'CO/LIST', 'SUB/LIST', 'CO/TEMPORAL', 'CO/DISJUNCTION', 'SUB/TEMPORAL', 'CO/PURPOSE', 'SUB/PURPOSE', 'CO/RESULT', 'SUB/RESULT', 'CO/CLAUSE', 'SUB/CLAUSE', 'CO/CONTRAST', 'SUB/CONTRAST', 'SUB/DISJUNCTION', "CO/LSIT", 'SUB/ATTRIBUTION', 'CO/ATTRIBUTION', 'SUB/SPATIAL', 'SUB/BACKGROUND', 'SUB/CAUSE']
    df = pd.DataFrame(rel, columns=['Relation'])
    df['Precision'] = precision
    df['Recall'] = recall
    df['F1_score'] = f1_score
    df.to_csv(sys.argv[4])
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("F1 Score: ", f1_score)