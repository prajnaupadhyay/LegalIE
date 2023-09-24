# LegalIE

Python 3.10 is needed to run the code.

`conda create -n legalIE python=3.10`

`conda activate legalIE`

## Dataset

Coordination Datasets are located under `data/CoordinationDataSet`. There are 3 folders:

#### 1) `gold`: This contains the gold data or the test data (terms can be interchangeably used). The same dataset is present in 4 formats:
##### `test.coord`: in the format of coordination trees(with \# for input)
###### `test_copy.coord`: in the format with Input: and Prediction: from `test.coord`
###### `test_mod.coord`: with some corrections for punctuation anamolies from `test_copy.coord`
###### `test_mod2.coord`: with `Preprocessor.get_mod2_file()` from `Utils/wire57.py` on `test.oord`
###### `test.conj`: in the format of split sentences
###### `test.labels`: in the format OpenIE uses (we do not use this format)
###### `test.txt`: in the format of plain text sentences

#### 2) `input`: all input files with the appropriate formats (contains `train.coord`)
#### 3) `output`: all output files with the appropriate formats

<!---
## Steps



**Dataset1** To obtain linearized output from Graphene's tree hierarchical sentence structure, run ProcessingDisSimTree.py.
Input: treeStructure.txt output:LinearizedTree.txt

**cmd:** python ProcessingDisSimTree.py output.txt

output: <br>
#Bell , based in Los Angeles , makes and distributes electronic , computer and building products . <br>
SUB/UNKNOWN_SUBORDINATION('Bell makes and distributes electronic , computer and building products .','Bell is based in Los Angeles .')

**Dataset2** To obtain coordiantion from OpenIE6, use preprocess.py file . It takes ptb_train_split_labels file as input and generates coordination tree file as output

Input file content : <br>

Seven Big Board stocks -- UAL , AMR , BankAmerica , Walt Disney , Capital Cities\/ABC , Philip Morris and Pacific Telesis Group -- stopped trading and never resumed . <br>
NONE NONE NONE NONE NONE CP_START SEP CP SEP CP SEP CP CP SEP CP CP SEP CP CP CC CP CP CP NONE CP_START CP CC CP CP NONE <br>
NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE <br>
NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE <br>

Output: <br>

#Seven Big Board stocks -- UAL , AMR , BankAmerica , Walt Disney , Capital Cities\/ABC , Philip Morris and Pacific Telesis Group -- stopped trading and never resumed . <br>
COORDINATION(" Seven Big Board stocks -- UAL -- stopped trading ." , " Seven Big Board stocks -- UAL -- never resumed ." , " Seven Big Board stocks -- AMR -- stopped trading ." , " Seven Big Board stocks -- AMR -- never resumed ." , " Seven Big Board stocks -- BankAmerica -- stopped trading ." , " Seven Big Board stocks -- BankAmerica -- never resumed ." , " Seven Big Board stocks -- Walt Disney -- stopped trading ." , " Seven Big Board stocks -- Walt Disney -- never resumed ." , " Seven Big Board stocks -- Capital Cities\/ABC -- stopped trading ." , " Seven Big Board stocks -- Capital Cities\/ABC -- never resumed ." , " Seven Big Board stocks -- Philip Morris -- stopped trading ." , " Seven Big Board stocks -- Philip Morris -- never resumed ." , " Seven Big Board stocks -- Pacific Telesis Group -- stopped trading ." , " Seven Big Board stocks -- Pacific Telesis Group -- never resumed ." ) <br>

-->

