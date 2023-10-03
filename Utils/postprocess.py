import sys
import spacy

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
        
        # path1 = "data/CoordinationDataSet/input/train.coord"
        # path2 = "data/CoordinationDataSet/input/train_copy.coord"
        
        for i in range(5):
            path1 = f"data/SubordinationDataSet/output/Graphene_Level{i}.coords"
            path2 = f"data/SubordinationDataSet/output/Graphene_Level{i}_IP.coord"

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
            if predictions[i].startswith("Input:"):
                predictions[i] = predictions[i].replace("Input: ", "")[:-1]
                predictions[i] =  " ".join([sent.text for sent in nlp(predictions[i])])
                fw.write("Input: " + predictions[i] + "\n")
            if predictions[i].startswith("Prediction:"):
                predictions[i] = predictions[i].replace("Prediction: ", "")[:-1]
                predictions[i] =  " ".join([sent.text for sent in nlp(predictions[i])]) 
                predictions[i] = predictions[i].replace("COORDINATION ( \"", "COORDINATION(\"")
                predictions[i] = predictions[i].replace(". \"", ".\"")
                predictions[i] = predictions[i].replace(" \\ ", "\\/")
                predictions[i] = predictions[i].replace(" - ", "-")
                predictions[i] = predictions[i].replace(") )", "))")
                fw.write("Prediction: " + predictions[i] + "\n\n")
        fr.close()
        fw.close()
        
if __name__ == "__main__":
    # Preprocessor.get_mod2_file()
    PostProcessor.preprocess_SubordData()
    # Preprocessor.get_copy_file()
    
    # if (len(sys.argv) != 3):
    #     print("Usage: python3 postprocess.py <input_file> <output_file>")
    #     exit(0)
    # PostProcessor.get_in_openie_format(sys.argv[1], sys.argv[2])