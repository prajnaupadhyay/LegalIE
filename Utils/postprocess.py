import sys
import spacy
import pandas as pd


class PostProcessor:

    @classmethod
    def get_copy_file(cls):
        path1 = "data/CoordinationDataSet/output2/predictions/Prediction_T5_large_test_copy.coord"
        path2 = "data/CoordinationDataSet/output2/predictions/Prediction_T5_large_test_copy2.coord"

        fr = open(path1, "r")
        fw = open(path2, "w")
        f = fr.readlines()

        for i in range(len(f)):
            # if i%3 == 0:
            #     f[i] = "Input: " + f[i][1:]
            # print(f[i])
            if i % 3 == 1:
                f[i] = "Prediction: " + f[i]
            fw.write(f[i])
        fr.close()
        fw.close()

    @classmethod
    def get_mod2_file(cls):
        # path1 = "data/CoordinationDataSet/gold/test.coord"
        # path2 = "data/CoordinationDataSet/gold/test_mod2.coord"

        path1 = "data/CoordinationDataSet/input/train.coord"
        path2 = "data/CoordinationDataSet/input/train_mod2.coord"

        fr = open(path1, "r")
        fw = open(path2, "w")
        f = fr.readlines()

        for i in range(len(f)):
            if i % 3 == 0:
                f[i] = "Input: " + f[i][1:]
                # print(f[i])
            if i % 3 == 1:
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

        path1 = "data/SubordinationDataSet/input/train_new.txt"
        path2 = "data/SubordinationDataSet/input/train_new_IP.txt"

        # for i in range(5):
        #     path1 = f"data/SubordinationDataSet/output/Graphene_Level{i}.coords"
        #     path2 = f"data/SubordinationDataSet/output/Graphene_Level{i}_IP.coord"

        fr = open(path1, "r")
        fw = open(path2, "w")
        f = fr.readlines()

        for i in range(len(f)):
            if i % 2 == 0:
                f[i] = "Input: " + f[i][1:]
                # print(f[i])
            if i % 2 == 1:
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
            if predictions[i].startswith("Input:"):
                predictions[i] = predictions[i].replace("Input: ", "")[:-1]
                predictions[i] = " ".join(
                    [sent.text for sent in nlp(predictions[i])])
                fw.write("Input: " + predictions[i] + "\n")
            if predictions[i].startswith("Prediction:"):
                predictions[i] = predictions[i].replace(
                    "Prediction: ", "")[:-1]
                predictions[i] = " ".join(
                    [sent.text for sent in nlp(predictions[i])])
                predictions[i] = predictions[i].replace(
                    "COORDINATION ( \"", "COORDINATION(\"")
                predictions[i] = predictions[i].replace(". \"", ".\"")
                predictions[i] = predictions[i].replace(" \\ ", "\\/")
                predictions[i] = predictions[i].replace(" - ", "-")
                predictions[i] = predictions[i].replace(") )", "))")
                fw.write("Prediction: " + predictions[i] + "\n\n")
        fr.close()
        fw.close()

    @classmethod
    def get_train_levels(cls):
        path1 = "data/CoordinationDataSet/input/train_copy.coord"

        fr = open(path1, "r")
        f = fr.readlines()
        l = [0, 0, 0, 0, 0, 0, 0, 0]
        for line in f:
            if line.startswith("Prediction: "):
                l[line.count("COORDINATION")] += 1
        print(l)

    @classmethod
    def get_label_numbers(cls):
        path2 = "data/SubordinationDataSet/gold/test_reduced_IP.coord"
        path1 = "data/SubordinationDataSet/input/train_IP.txt"

        f1 = open(path1, "r").read()
        f2 = open(path2, "r").read()
        # f = fr.readlines()
        rel = ['CO/ELABORATION', 'SUB/ELABORATION', 'CO/CONDITION', 'SUB/CONDITION', 'CO/LIST', 'SUB/LIST', 'CO/TEMPORAL', 'CO/DISJUNCTION', 'SUB/TEMPORAL', 'CO/PURPOSE', 'SUB/PURPOSE', 'CO/RESULT',
               'SUB/RESULT', 'CO/CLAUSE', 'SUB/CLAUSE', 'CO/CONTRAST', 'SUB/CONTRAST', 'SUB/DISJUNCTION', "CO/LSIT", 'SUB/ATTRIBUTION', 'CO/ATTRIBUTION', 'SUB/SPATIAL', 'SUB/BACKGROUND', 'SUB/CAUSE']
        df = pd.DataFrame(rel, columns=['Relation'])
        rel_num1 = {i: f1.count(i) for i in rel}
        rel_num2 = {i: f2.count(i) for i in rel}
        df['Train'] = rel_num1.values()
        df['Test'] = rel_num2.values()
        df.to_csv("sub_stat.csv")
        print("Train\n", rel_num1)
        print("Test\n", rel_num2)

    @classmethod
    def make_subord(cls):
        rel = ['CO/ELABORATION', 'SUB/BACKGROUND', 'SUB/ELABORATION', 'CO/LIST', 'CO/LSIT', 'SUB/ATTRIBUTION', 'CO/CONTRAST',
               'CO/DISJUNCTION', 'SUB/SPATIAL', 'SUB/PURPOSE', 'SUB/CONDITION', 'SUB/CAUSE', 'SUB/TEMPORAL', 'SUB/RESULT', 'SUB/CONTRAST']

        # path1 = "data/SubordinationDataSet/input/train_IP.txt"
        # path2 = "data/SubordinationDataSet/input/train_subord_IP.txt"

        path1 = "data/SubordinationDataSet/gold/test_reduced_IP.txt"
        path2 = "data/SubordinationDataSet/gold/test_reduced_subord_IP.txt"

        fr = open(path1, "r")
        fw = open(path2, "w")
        f = fr.readlines()
        for line in f:
            if line.startswith("Prediction: "):
                for r in rel:
                    line = line.replace(r, "SUBORDINATION")
                fw.write(line)
            else:
                fw.write(line)

    @classmethod
    def make_CoSubOrd(cls):
        relCo = ['CO/ELABORATION', 'CO/LIST', 'CO/LSIT',
                 'CO/CONTRAST', 'CO/DISJUNCTION', ]
        relSub = ['SUB/BACKGROUND', 'SUB/ELABORATION', 'SUB/ATTRIBUTION', 'SUB/SPATIAL',
                  'SUB/PURPOSE', 'SUB/CONDITION', 'SUB/CAUSE', 'SUB/TEMPORAL', 'SUB/RESULT', 'SUB/CONTRAST']

        path1 = "data/SubordinationDataSet/input/train_IP.txt"
        path2 = "data/SubordinationDataSet/input/train_cosubord_IP.txt"

        # path1 = "data/SubordinationDataSet/gold/test_reduced_IP.txt"
        # path2 = "data/SubordinationDataSet/gold/test_reduced_cosubord_IP.txt"

        fr = open(path1, "r")
        fw = open(path2, "w")
        f = fr.readlines()
        for line in f:
            if line.startswith("Prediction: "):
                for r in relSub:
                    line = line.replace(r, "SUBORDINATION")
                for r in relCo:
                    line = line.replace(r, "COORDINATION")
                fw.write(line)
            else:
                fw.write(line)

    @classmethod
    def LIDC_formater(cls):
        path1 = "data/SubordinationDataSet/gold/LIDC_test_raw.txt"
        path2 = "data/SubordinationDataSet/gold/LIDC_test_IP.txt"

        fr = open(path1, "r")
        fw = open(path2, "w")
        predictions = fr.readlines()
        nlp = spacy.load('en_core_web_sm')

        for i in range(len(predictions)):
            fw.write("Input: " + predictions[i][:-1] + "\n")
            fw.write("Prediction: " + predictions[25][:-1] + "\n\n")
        fr.close()
        fw.close()

    @classmethod
    def ablation_study1(cls):
        r2 = ['CO/ELABORATION', 'SUB/ELABORATION', 'CO/CONDITION', 'SUB/CONDITION', 'CO/LIST', 'SUB/LIST', 'CO/TEMPORAL', 'CO/DISJUNCTION', 'SUB/TEMPORAL', 'CO/PURPOSE', 'SUB/PURPOSE', 'CO/RESULT', 'SUB/RESULT',
              'CO/CLAUSE', 'SUB/CLAUSE', 'CO/CONTRAST', 'SUB/CONTRAST', 'SUB/DISJUNCTION', "CO/LSIT", 'SUB/ATTRIBUTION', 'CO/ATTRIBUTION', 'SUB/SPATIAL', 'SUB/BACKGROUND', "SUB/CAUSE", "SUB / ELABORATION"]

        path1 = "data/SubordinationDataSet/input/train_IP.txt"
        path2 = "data/SubordinationDataSet/input/train_Level0+1_IP.txt"
        path3 = "data/SubordinationDataSet/input/train_Level0+2+3+4_IP.txt"

        fr = open(path1, "r")
        fw1 = open(path2, "w")
        fw2 = open(path3, "w")
        f = fr.readlines()
        alt = True
        for i in range(len(f)):
            if f[i].startswith("Prediction: "):
                count = 0
                for r in r2:
                    count += f[i].count(r)
                if count == 0:
                    if alt:
                        fw1.write(f[i-1])
                        fw1.write(f[i] + '\n')
                    else:
                        fw2.write(f[i-1])
                        fw2.write(f[i] + '\n')
                    alt = not alt
                elif count == 1:
                    fw1.write(f[i-1])
                    fw1.write(f[i] + '\n')
                elif count == 2:
                    fw2.write(f[i-1])
                    fw2.write(f[i] + '\n')
                elif count == 3:
                    fw2.write(f[i-1])
                    fw2.write(f[i] + '\n')
                elif count == 4:
                    fw2.write(f[i-1])
                    fw2.write(f[i] + '\n')
                else:
                    fw2.write(f[i-1])
                    fw2.write(f[i] + '\n')
        fr.close()
        fw1.close()
        fw2.close()


if __name__ == "__main__":
    # Preprocessor.get_mod2_file()
    # PostProcessor.preprocess_SubordData()
    # PostProcessor.make_CoSubOrd()
    # Preprocessor.get_copy_file()
    # PostProcessor.get_train_levels()
    # PostProcessor.get_label_numbers()
    # PostProcessor.LIDC_formater()
    PostProcessor.ablation_study1()
    # if (len(sys.argv) != 3):
    #     print("Usage: python3 postprocess.py <input_file> <output_file>")
    #     exit(0)
    # PostProcessor.get_in_openie_format(sys.argv[1], sys.argv[2])
