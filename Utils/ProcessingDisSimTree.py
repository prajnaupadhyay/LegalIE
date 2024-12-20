# -*- coding: utf-8 -*-
"""ProcessingTree.ipynb

Automatically generated by Colaboratory.

Original file is located at
"""

inputFile = "validTree.txt"
middleFile= "test.txt"
out = open("BART_Brackets.txt","w")
num_brackets = 0

with open(inputFile, 'r') as input_file, open(middleFile, 'w') as mid_file:
    # Read the contents of the input file
    text = input_file.read()

    # Split the text into lines
    lines = text.split('\n')

    previous_count = 0
    modified_lines = []

    # Count the '|' characters in each line and add closing bracket if count is decreasing
    for line in lines:
        count = line.count('|')

        if count < previous_count :
            closing_brackets = previous_count - count
            closing_bracket = ')' * closing_brackets
            modified_lines[-1] += ' ' + closing_bracket + '\n' + line
        else:
            modified_lines.append(line)

        previous_count = count

    # Write the modified lines to the output file
    for line in modified_lines:
        mid_file.write(line + '\n')



with open(middleFile,'r') as file:
    for line in file:
        if line.startswith('#'):
            while((num_brackets) > 0):
                out.write(")")
                num_brackets = num_brackets -1


            out.write("\n"+line.strip()+"\n")
        str1 ="─>"
        if str1 in line:
            line=line.strip().replace(str1, " ")
            line= line.strip().replace("|", "")
            print("Line:",line)
            if "SUB" in line or "CO" in line:
                line = line.split()
                print("SPlit Line", line[0])
                out.write(line[0]+"(")
                num_brackets = num_brackets+1
            else:
                out.write(line.strip())
                if(")" in line):
                    num_brackets = num_brackets -1

    while((num_brackets) > 0):
                out.write(")")
                num_brackets = num_brackets -1



import re

def add_comma_after_inverted_commas(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file:
        content = input_file.read()

    # Use regular expression to find closing inverted commas followed by a space
    pattern1 = r"''"
    pattern2 = r"'S"
    pattern3 = r"'C"
    pattern4 = r"\)'"
    replaced_content1 = re.sub(pattern1, "','", content)
    replaced_content2 = re.sub(pattern2, "', S", replaced_content1)
    replaced_content3 = re.sub(pattern3, "', C", replaced_content2)
    replaced_content4 = re.sub(pattern4, "), '", replaced_content3)



    lines = replaced_content4.split("\n")  # Split replaced content into lines

    modified_lines = []  # List to store modified lines

    for line in lines:
        if not line.startswith("#"):
          if line.strip():
            if re.search(r"(CO/|SUB/)", line):  # Check if the line contains "CO/" or "SUB/"
                modified_lines.append(line)
            else:
                modified_lines.append("NONE")
        else:
            modified_lines.append(line)

    modified_content = "\n".join(modified_lines)  # Join modified lines into a string

    with open(output_file_path, 'w') as output_file:
        output_file.write(modified_content)


input_file_path = 'BART_Brackets.txt'
output_file_path = 'valid.txt'
add_comma_after_inverted_commas(input_file_path, output_file_path)
