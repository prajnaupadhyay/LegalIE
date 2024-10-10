# LegalIE

#### 1) Create a conda environment
Python 3.10 is needed to run the code.

```
conda create -n legalIE python=3.10


conda activate legalIE
```
#### 2) Install requirements 

```
pip install -r requirements.txt
```
#### 3) Execute run.py in the below format

```
python run.py train-test/train/test train.txt model_dir test.txt predictions.txt Model_name batch_size seed
```

##### `train-test` - both for training the model and testing it on test sentences
 
##### `train` - only trains the model
 
##### `test` - only tests the model

##### `train.txt` - training dataset
 
##### `model_dir` - directory where the model would be saved 
 
##### `test.txt` - test file 
 
##### `predictions.txt` - the name of the prediction file 
 
##### `Model_name` - T5 or BART

       


## Dataset

Datasets are located under `data/`. There are 3 folders:

#### 1) `gold`: This contains the gold data or the test data (terms can be interchangeably used). The same dataset is present in 4 formats:
###### `test.coord`: in the format of coordination trees
###### `test.conj`: in the format of split sentences
###### `test.labels`: in the format OpenIE uses (we do not use this format)
###### `test.txt`: in the format of plain text sentences

#### 2) `input`: all input files with the appropriate formats (contains `train.coord`)
#### 3) `output`: all output files with the appropriate formats

## Models

##### `T5 Base`: https://huggingface.co/bphclegalie/t5-base-legen <br>
##### `T5 Small`: https://huggingface.co/bphclegalie/t5-small-custom-loss

