
inputFile = "CARB_SEntences/openIETRainConj/testSentPredictions.txt.conj"
output = open("TestCOORDIANTIONTRee1.txt","w")

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
            output.write("COORDINATION( "+line+",")
            num_brackets = num_brackets+1
        elif num_brackets==0 and line == "\n" and i=='labels':
            output.write("None\n")
            i='input'
        elif line == "\n":
            output.write(")"*num_brackets+"\n")
            num_brackets = 0
            i='input'


#Replaces two COORDINATION with one
from more_itertools import locate
import numpy as np
output = open("TestFinalCordinationTree.txt","w")
inputFile = "TestCOORDIANTIONTRee1.txt"
def replace_two_coordination(inputFile, output):
    with open(inputFile,"r") as file:
        for line in file:
            if line.startswith('#'):
                output.write("\n"+line)
            
        
        
            else:
                #indices =[]
                #print(type(line))
                count =0
                line_words = line.split()
                #print(line_words)
                for words in line_words:
                    if "COORDINATION(" in words:
                        count = count+1
                
            
                if(count ==2):
                    index = line.rfind("COORDINATION", 0, len(line))
                    last_character = index+13
                    line = line[0:index-1] + ", "+line[last_character:]
            
                    output.write(line[:-2])
                
                else:
                    output.write(line)
            
            

                        
               
