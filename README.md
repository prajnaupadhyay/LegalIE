# LegalIE

Python 3.10 is needed to run the code.

`conda create -n legalIE python=3.10`

`conda activate legalIE`


**Dataset1** To obtain linearized output from Graphene's tree hierarchical sentence structure, run ProcessingDisSimTree.py.
Input: treeStructure.txt output:LinearizedTree.txt

**cmd:** python ProcessingDisSimTree.py output.txt

output: <br>
#Bell , based in Los Angeles , makes and distributes electronic , computer and building products . <br>
SUB/UNKNOWN_SUBORDINATION('Bell makes and distributes electronic , computer and building products .','Bell is based in Los Angeles .')

**Dataset2 ** To obtain coordiantion from OpenIE6, use preprocess.py file . It takes ptb_train_split_labels file as input and generates coordiantion tree file as output

Input file content : <br>

Seven Big Board stocks -- UAL , AMR , BankAmerica , Walt Disney , Capital Cities\/ABC , Philip Morris and Pacific Telesis Group -- stopped trading and never resumed . <br>
NONE NONE NONE NONE NONE CP_START SEP CP SEP CP SEP CP CP SEP CP CP SEP CP CP CC CP CP CP NONE CP_START CP CC CP CP NONE <br>
NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE <br>
NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE <br>

Output: <br>

#Seven Big Board stocks -- UAL , AMR , BankAmerica , Walt Disney , Capital Cities\/ABC , Philip Morris and Pacific Telesis Group -- stopped trading and never resumed . <br>
COORDINATION(" Seven Big Board stocks -- UAL -- stopped trading ." , " Seven Big Board stocks -- UAL -- never resumed ." , " Seven Big Board stocks -- AMR -- stopped trading ." , " Seven Big Board stocks -- AMR -- never resumed ." , " Seven Big Board stocks -- BankAmerica -- stopped trading ." , " Seven Big Board stocks -- BankAmerica -- never resumed ." , " Seven Big Board stocks -- Walt Disney -- stopped trading ." , " Seven Big Board stocks -- Walt Disney -- never resumed ." , " Seven Big Board stocks -- Capital Cities\/ABC -- stopped trading ." , " Seven Big Board stocks -- Capital Cities\/ABC -- never resumed ." , " Seven Big Board stocks -- Philip Morris -- stopped trading ." , " Seven Big Board stocks -- Philip Morris -- never resumed ." , " Seven Big Board stocks -- Pacific Telesis Group -- stopped trading ." , " Seven Big Board stocks -- Pacific Telesis Group -- never resumed ." ) <br>



