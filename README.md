# LegalIE

#### 1) Create conda environment
Python 3.10 is needed to run the code.

`conda create -n legalIE python=3.10`

`conda activate legalIE`
#### 2) Install requirements 
pip install -r requirements.txt

#### 3)python run.py train-test/train/test train.txt model_dir test.txt predictions.txt Model_name batch_size seed

### train-test - both for training the model and testing it on test sentences
### train- only trains the model
### test - only tests the model

### train.txt - training dataset
### model_dir - directory where model would be saved
### test.txt - test file
### predictions.txt - name of the prediction file
### Model_name - T5 or BART

       


## Dataset

Coordination Datasets are located under `data/CoordinationDataSet`. There are 3 folders:

#### 1) `gold`: This contains the gold data or the test data (terms can be interchangeably used). The same dataset is present in 4 formats:
###### `test.coord`: in the format of coordination trees
###### `test.conj`: in the format of split sentences
###### `test.labels`: in the format OpenIE uses (we do not use this format)
###### `test.txt`: in the format of plain text sentences

#### 2) `input`: all input files with the appropriate formats (contains `train.coord`)
#### 3) `output`: all output files with the appropriate formats

## Models

T5 Base: https://huggingface.co/bphclegalie/t5-base-legen <br>
T5 Small: https://huggingface.co/bphclegalie/t5-small-custom-loss


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


