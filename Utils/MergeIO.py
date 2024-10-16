#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Removes spl character and line starting with Testing
fileX = "TestNotFormatted.txt"
output1 = open("TestFlattenedTreeWithoutSplChar1.txt","w")
with open(fileX, "r") as file:
    for line in file:
        if line.startswith(" Testing"):
            if "COORDINATION(" in line:
                index = line.find("COORDINATION( ")
                #print(index)
            elif "NONE" in line:
                index = line.find("NONE")
            output1.write(line[index:])
        elif not line.startswith("Testing "):
            output1.write(line)
            
            #print(line[index:])


# In[6]:


#Merge two files
#file1 = "sampleinput.txt"
file1 = "testSent.txt"
#file2 = "sampleflatcoordsTree.txt"
file2 = "TestFlattenedTreeWithoutSplChar1.txt"
file3=open("TestmergeIO.txt", "w+")
with open(file1,"r") as f1, open(file2,"r") as f2:
    for line, coords in zip(f1,f2):
        #print("#"+line)
        file3.write("\n"+'#'+line)
        
        #for coords in f2:
        #print(coords)
        file3.write(coords+"\n")
            
        


# In[7]:


#Formatted MergedFile - Adds brackets
infile = "TestmergeIO.txt"
output = open("test.txt","w")
    
with open(infile,"r") as file:
    for line in file:
        if line.startswith('#') and len(line) > 3:
            #print(len(line),line)
            output.write(line)
        elif line.startswith("COORDINATION("):
            brackets = line.count("COORDINATION(", 0, len(line))
            line = line.replace("NONE", ")"*brackets)
            #print(line)
            output.write(line)
        

#Adds NONE between two consecutive lines starting with #
def add_none_between_lines(file_path):
    lines = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for i in range(len(lines)):
        line = lines[i]
        modified_lines.append(line.strip())
        if i < len(lines) - 1 and line.startswith('#') and lines[i + 1].startswith('#'):
            modified_lines.append('NONE')

    with open(file_path, 'w') as file:
        file.write('\n'.join(modified_lines))

if __name__ == "__main__":
    file_path = "test.txt"
    add_none_between_lines(file_path)
