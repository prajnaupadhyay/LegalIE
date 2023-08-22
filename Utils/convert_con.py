#Converts Conj file generated by openie6 in the format required for computing score
inputFile = "ptb-test_split_removedExtra_Predictions.txt.conj"
output = open("ConvertedConj.txt","w")

prevLine = " "
inputs= []
labels= []
num_brackets = 0
i='input'
with open(inputFile,"r") as file:
    for line in file:
        if i=='input' and line!='\n':
            line = line.strip()
            inputs.append(line)
            output.write("#"+line+"\n")
            #print(line)
            i='labels'
        elif i=='labels' and line  != "\n":
            line = line.strip()
            labels.append(line)
            output.write('\"'+line+'\"'+',')
            num_brackets = num_brackets+1
        elif num_brackets==0 and line == "\n" and i=='labels':
            output.write("None\n")
            i='input'
        elif line == "\n":
           # output.write(")"*num_brackets+"\n")
            output.write("\n")
            num_brackets = 0
            i='input'
