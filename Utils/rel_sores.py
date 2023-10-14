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
        ref, pred, inp = [], [], []
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

class rel_gen:
    @classmethod
    def reader(cls, model, ref_path, pred_path):
        ref_file = open(ref_path, "r").read()
        pred_file = open(pred_path, "r")
        # rel = ['CO/ELABORATION', 'SUB/ELABORATION', 'CO/LIST', 'CO/LSIT', 'SUB/ATTRIBUTION', 'CO/CONTRAST', 'CO/DISJUNCTION', 'SUB/SPATIAL', 'SUB/PURPOSE', 'SUB/CONDITION', 'SUB/CAUSE', 'SUB/TEMPORAL', 'SUB/BACKGROUND', 'SUB/RESULT', 'SUB/CONTRAST', 'SUB/LIST', 'CO/TEMPORAL', 'CO/PURPOSE', 'CO/RESULT',  'CO/CLAUSE', 'SUB/CLAUSE', 'SUB/DISJUNCTION', 'CO/ATTRIBUTION', 'CO/CONDITION']
        rel = ['SUBORDINATION', 'CO/ELABORATION', 'SUB/BACKGROUND', 'SUB/ELABORATION', 'CO/LIST', 'CO/LSIT', 'SUB/ATTRIBUTION', 'CO/CONTRAST', 'CO/DISJUNCTION', 'SUB/SPATIAL', 'SUB/PURPOSE', 'SUB/CONDITION', 'SUB/CAUSE', 'SUB/TEMPORAL', 'SUB/RESULT', 'SUB/CONTRAST']
        ref, ref_org = [], []
        pred, pred_org = [], []
        inp = []
        count = np.zeros((ref_file.count("Prediction: "), len(rel)), np.int8)
        c = 0
        ref_file = open(ref_path, "r")
        for j, line in enumerate(ref_file):
            if line.startswith("Prediction: "):
                prediction = line.replace("Prediction: ", "")[:-1]
                count[c] = np.array([1 if i in prediction else 0 for i in rel ])
                c += 1
                new_sentences = sorted(get_sentences_from_tree_labels(model, prediction))
                sentences = [s.strip() for s in new_sentences if s.strip() != ""]
                ref.append(sentences)
                ref_org.append(prediction)
            if line.startswith("Input: "):
                input = line.replace("Input: ", "")[:-1]
                inp.append(input)
                
        for line in pred_file.readlines():
            if line.startswith("Prediction: "):
                prediction = line.replace("Prediction: ", "")[:-1]
                new_sentences = sorted(get_sentences_from_tree_labels(model, prediction))
                sentences = [s.strip() for s in new_sentences  if s.strip() != ""]
                pred.append(sentences)
                pred_org.append(prediction)
                
        if len(ref) != len(pred):
            raise Exception(f"Number of sentences in reference {len(ref)} and prediction {len(pred)} are not equal")
        return ref, pred, count, ref_org, pred_org, inp
    
    @classmethod
    def writer(cls, model, ref_path, pred_path, path, org_path):
        ref, pred, count, ref_org, pred_org, inp = cls.reader(model, ref_path, pred_path)
        # print(inp)
        rel_exp = [[] for i in range(count.shape[1])]
        rel_exp_org = [[] for i in range(count.shape[1])]
        inp_exp = [[] for i in range(count.shape[1])]
        # print(count.shape[1])
        # print(len(rel_exp))
        for i in range(count.shape[1]):
            _t = np.nonzero(count[:,i])
            _t = np.ndarray.tolist(_t[0])
            # print("Printing", _t)
            for j in range(len(_t)):
                rel_exp[i].append(ref[_t[j]])
                rel_exp_org[i].append(ref_org[_t[j]])
                # rel_exp[i].append(pred[_t[j]])
                # rel_exp_org[i].append(pred_org[_t[j]])
                inp_exp[i].append(inp[_t[j]])
        # print(inp_exp)
        fw1 = open(path, "w")
        fw2 = open(org_path, "w")
        for i in range(len(rel_exp)):
            for j in range(len(rel_exp[i])):
                fw1.write(f"{inp_exp[i][j]}\n")
                fw2.write(f"Input: {inp_exp[i][j]}\n")
                fw2.write(f"Prediction: {rel_exp_org[i][j]}\n")
                for k in range(len(rel_exp[i][j])):
                    fw1.write(f"{rel_exp[i][j][k]}\n")
                fw1.write("\n")
                fw2.write("\n")        
    
    
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 wire.py T5/BART/OpenIE <refernce_file> <prediction_file> <result_csv>")
        exit(0)
    # precision, recall, f1_score =  rel_overlap_scorer.scorer(sys.argv[1], sys.argv[2], sys.argv[3])
    # rel = ['CO/ELABORATION', 'SUB/ELABORATION', 'CO/CONDITION', 'SUB/CONDITION', 'CO/LIST', 'SUB/LIST', 'CO/TEMPORAL', 'CO/DISJUNCTION', 'SUB/TEMPORAL', 'CO/PURPOSE', 'SUB/PURPOSE', 'CO/RESULT', 'SUB/RESULT', 'CO/CLAUSE', 'SUB/CLAUSE', 'CO/CONTRAST', 'SUB/CONTRAST', 'SUB/DISJUNCTION', "CO/LSIT", 'SUB/ATTRIBUTION', 'CO/ATTRIBUTION', 'SUB/SPATIAL', 'SUB/BACKGROUND', 'SUB/CAUSE']
    # df = pd.DataFrame(rel, columns=['Relation'])
    # df['Precision'] = precision
    # df['Recall'] = recall
    # df['F1_score'] = f1_score
    # df.to_csv(sys.argv[4])
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("F1 Score: ", f1_score)
    
    rel_gen.writer(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    
    