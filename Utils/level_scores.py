from wire57 import wire57_scorer
from preprocess import get_sentences_from_tree_labels
import sys


class wire57_scorer_level(wire57_scorer):
    @classmethod
    def reader(cls, model, ref_path, pred_path):
        ref_file = open(ref_path, "r")
        pred_file = open(pred_path, "r")
        ref = []
        pred = []
        count = []
        for line in ref_file:
            if line.startswith("Prediction: "):
                prediction = line.replace("Prediction: ", "")[:-1]
                count.append(prediction.count("COORDINATION"))
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
        refs = [[] for i in range(max(count)+1)]
        preds = [[] for i in range(max(count)+1)]
        for i in range(len(count)):
            refs[count[i]].append(ref[i])
            preds[count[i]].append(pred[i])
        return refs, preds
    
    @classmethod
    def scorer(cls, model, ref_path, pred_path):
        refs, preds = cls.reader(model, ref_path, pred_path)
        for j in range(len(refs)):
            precision, recall, f1_score, f1 = 0, 0, 0, 0
            for i in range(len(refs[j])):
                p, r = cls.matcher_using_f1(refs[j][i], preds[j][i])
                # print(p, r)
                if(p != 0 and r != 0):
                    f1 = 2*p*r/(p+r)
                else:
                    f1 = 0
                precision += p
                recall += r
                f1_score += f1
            Pr, Rc, F1s = precision/len(refs[j]), recall/len(refs[j]), f1_score/len(refs[j])
            print(f"Level {j} with count {len(refs[j])}")
            print("Precision: ", Pr)
            print("Recall: ", Rc)
            print("F1 Score: ", F1s, "\n")
            
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 wire57.py T5/BART/OpenIE <refernce_file> <prediction_file>")
        exit(0)
    wire57_scorer_level.scorer(sys.argv[1], sys.argv[2], sys.argv[3])             