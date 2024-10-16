from preprocess import get_sentences_from_tree_labels
import numpy as np
import sys
from scipy.optimize import linear_sum_assignment
import re


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
                new_sentences = sorted(
                    get_sentences_from_tree_labels(model, prediction))
                # print([s.strip() for s in new_sentences if s.strip() != ""])
                sentences = [set((s.strip()).split())
                             for s in new_sentences if s.strip() != ""]
                # print(sentences)
                ref.append(sentences)

        if model == "OpenIE":
            pred = pred_file.read().split('\n\n')
            for i in range(len(pred)):
                # print(pred[i])
                _t = [s.strip()
                      for s in pred[i].splitlines() if s.strip() != ""]
                # _t = _t[1:]
                # print(_t)
                pred[i] = [set((s.strip()).split()) for s in sorted(_t[1:])]
                # print(pred[i])
            pred = pred[:-1]
        else:
            for line in pred_file.readlines():
                if line.startswith("Prediction: "):
                    prediction = line.replace("Prediction: ", "")[:-1]
                    new_sentences = sorted(
                        get_sentences_from_tree_labels(model, prediction))
                    sentences = [set((s.strip()).split())
                                 for s in new_sentences if s.strip() != ""]
                    # print(sentences)
                    pred.append(sentences)
        if len(ref) != len(pred):
            raise Exception(
                f"Number of sentences in reference {len(ref)} and prediction {len(pred)} are not equal")
        return ref, pred

    @classmethod
    def matcher_using_jaccard(cls, ref_set, pred_set):
        if (len(ref_set) == 0 and len(pred_set) == 0):
            return 1.0, 1.0
        elif (len(ref_set) == 0 or len(pred_set) == 0):
            return 0.0, 0.0
        mat = np.zeros((len(pred_set), len(ref_set)), dtype=np.float16)
        for i in range(len(pred_set)):
            for j in range(len(ref_set)):
                mat[i][j] = float(len(pred_set[i].intersection(
                    ref_set[j]))) / len(pred_set[i].union(ref_set[j]))
        # row_ind, col_ind = None, None
        row_ind, col_ind = linear_sum_assignment(mat, maximize=True)
        # print(row_ind, col_ind)
        intsec = [len(pred_set[i].intersection(ref_set[j]))
                  for i, j in zip(row_ind, col_ind)]
        precision = [float(intsec[i])/len(pred_set[i])
                     for i in range(len(intsec))]

        row_ind, col_ind = linear_sum_assignment(mat.T, maximize=True)
        # print(row_ind, col_ind)
        intsec = [len(ref_set[i].intersection(pred_set[j]))
                  for i, j in zip(row_ind, col_ind)]
        recall = [float(intsec[i])/len(ref_set[i]) for i in range(len(intsec))]

        return sum(precision)/len(pred_set), sum(recall)/len(ref_set)

    @classmethod
    def matcher_using_f1(cls, ref_set, pred_set):
        if (len(ref_set) == 0 and len(pred_set) == 0):
            return 1.0, 1.0
        # elif (len(ref_set) == 0 and pred_set[0] == pred_set[-1]):
        #     return 1.0, 1.0
        elif (len(ref_set) == 0 or len(pred_set) == 0):
            return 0.0, 0.0

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
        # print("Precision\n", precision)
        # print("Recall\n", recall)
        # print("F1 scores\n", f1_score)
        row_ind, col_ind = linear_sum_assignment(f1_score, maximize=True)
        # print(precision[row_ind, col_ind], precision[row_ind, col_ind].sum())
        # print(recall[row_ind, col_ind], recall[row_ind, col_ind].sum())
        # print(len(pred_set), len(ref_set))
        return (precision[row_ind, col_ind].sum())/len(pred_set), (recall[row_ind, col_ind].sum())/len(ref_set)

    @classmethod
    def scorer(cls, model, ref_path, pred_path):
        ref, pred = cls.reader(model, ref_path, pred_path)
        precision, recall, f1_score, f1 = 0, 0, 0, 0
        for i in range(len(ref)):
            # print(3*i+1)
            p, r = cls.matcher_using_f1(ref[i], pred[i])
            # print(p, r)
            if (p != 0 and r != 0):
                f1 = 2*p*r/(p+r)
            else:
                f1 = 0
            precision += p
            recall += r
            f1_score += f1
        return precision/len(ref), recall/len(ref), f1_score/len(ref)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 wire57.py T5/BART/OpenIE <refernce_file> <prediction_file>")
        exit(0)
    precision, recall, f1_score = wire57_scorer.scorer(
        sys.argv[1], sys.argv[2], sys.argv[3])
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1_score)
