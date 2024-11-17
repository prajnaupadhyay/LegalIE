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

### 3) Training and Prediction

``` 
python run.py train-test <training_data_file> <model_directory> <test_data_file> <prediction_file> <model_name> <batch_size> <seed>
```

1. You can only use `train` or `test` instead of `train-test` if you want to do these steps individually.

2. `<training_data_file>` is are located in the `input/` folder of datasets which are located in `data/`.

3. `<model_directory>` is the location where the fine-tuned model will be saved after training and the re-trained (and finetuned) model located here will be used to generate predictions during testing.

4. `<test_data_file>` are located in the `gold/` folder of datasets which are located in `data/`.

5. `<prediction_file>` is the path where predictions generated by our model will be saved.

6. `<model_name>` will be `T5` or `BART`. The size of the model can changed through code.

7. `<batch_size>` is the batch size used during training. It can be $3, 16, 24, 32$ but $3$ gave the best results in most cases.

8. `<seed>` is a random seed value used for initialization. (We averaged results across 3 runs, this was used for that purpose.)

Please note that `<training_data_file>`, `<model_directory>`, `<test_data_file>` and `<prediction_file>` are relative paths and not just names.

### Evaluation

``` 
python Utils/wire57.py <model_name> <referance_file> <prediction_file>
```
1. `<model_name>` is the model used to generate `<prediction_file>`. It can be `T5` or `BART`

2. `<referance_file>` is a file with ground truth for the test dataset. These are located in `gold\` folder of each dataset which are in `data\`

3. `<prediction_file` is a file containing outputs generated by the model. Scores of this file will be calculated.
Please note that `<referance_file>` and `<prediction_file>` are relative paths and not just file names.

## Dataset
```
 data
 |-CoordinationDataSet 
 | |-gold
 | | |-CARB_test.txt (CARB dataset sentences. Model trained on co-ordination was used to generate output.)
 | | |-test.coord (Test file with old format)
 | | |-test.labels (Ground truth boudary labels for OpenIE6 model)
 | | |-test.txt (Raw test file without gold sentences)
 | | |-test_copy.coord (Test file in correct format. Use this file while running the code.)
 | | |-test_generative.conj (Test file where sentences from tree are seperated converted into a list. This file was used for manual comparion with output generated by T5 or BART)
 | | |-test_mod.coord (Modified version of test.coord)
 | | |-test_mod2.coord (Modified version of test_mod.coord)
 | | |-test_OpenIE6.conj (Test file where sentences from linearlized discourse tree are converted into a list. This file was used for manual comparion with output generated by OpenIE6)
 | |-input
 | | |-train.coord (Training file with old format)
 | | |-train_copy.coord (Training file in correct format. Use this file while running the code.)
 | |-output2 (BART-base were ran for 2 learning rates. File containing 'l1' with 10^-5 and other bart ones with 2*10^-5. Learning rate for T5 was 10^-5)
 | | |-evaluations
 | | | |-rouge
 | | | | |-Result_<model_name>_<model_size>_b<batch_size>_rouge.txt (Rough scores for generative models)
 | | | | |-Result_OpenIE.txt (Rough scores for OpenIE6)
 | | | |-wire57_f1
 | | | | |-Result_<model_name>_<model_size>_b<batch_size>_wire57v3.txt (Overlap scores for generative models)
 | | | | |-Result_OpenIE_wire57v3.txt (Overlap scores for OpenIE6)
 | | | |-wire57_f1_level
 | | | | |-Result_<model_name>_<model_size>_b<batch_size>.txt (Level-wise scores for generative models)
 | | | | |-Result_OpenIE_wire57v3.txt (Level-wise Overlap scores for OpenIE6)
 | | | | |-Result_T5_small_b03_pos_wire57v3.txt (Score for Parts-of-Speech taggging experiment)
 | | |-predictions
 | | | |-Prediction_<model_name>_<model_size>_b<batch_size>.conj (Generated predictions are converted into list format)
 | | | |-Prediction_<model_name>_<model_size>_b<batch_size>.coord (Predictions genrated by model in linearized discourse tree format)
 | | | |-Prediction_T5_small_b03_pos.coord (Parts-of-Speech taggging experiment)
 | | | |-CARB_Prediction_T5_base_b03.coord (Prediction for CARB sentences. Model was trained on Coordination sentences)
 | |-outputv2 (Contains repetition of experiments in output2 with different seed and BART-small with 3 different seed)
 | | |-evaluations
 | | | |-rouge
 | | | | |-Result_<model_name>_<model_size>_b<batch_size>_v<exp_no>_rouge.txt
 | | | |-wire57_f1
 | | | | |-Result_<model_name>_<model_size>_b<batch_size>_v<exp_no>_wire57v3.txt
 | | | | |-Result_T5_small_b03_ed_e_wire57v2.txt (Ablation Study: Encoder parameters frozen)
 | | | | |-Result_T5_small_b03_ed_d_wire57v2.txt (Ablation Study: Decoder parameters frozen)
 | | | |-wire57_f1_level
 | | | | |-Result_<model_name>_<model_size>_b<batch_size>_v<exp_no>_wire57v3
 | | |-predictions
 | | | |-Predictions_<model_name>_<model_size>_b<batch_size>_v<exp_no>.txt
 | | | |-Predictions_T5_small_b03_ed_e.txt (Ablation Study: Encoder parameters frozen)
 | | | |-Predictions_T5_small_b03_ed_d.txt (Ablation Study: Decoder parameters frozen)
 |-SubordinationDataSet (All files are from Subordination dataset on this directoru unless mentioned otherwise)
 | |-gold (Use files ending with IP when running the code)
 | | |-LIDC_test_IP.txt (LIDC dataset file in correct format.)
 | | |-LIDC_test_raw.txt (LIDC dataset file in raw format)
 | | |-test.txt (Test file in raw format)
 | | |-test_IP.txt (Test file in correct format)
 | | |-test_reduced.txt (Test file with reduced sentences in raw format)
 | | |-test_reduced_IP.txt (Test file with reduced sentences in correct format)
 | | |-test_reduced_cosubord_IP.txt (Test file with reduced sentences where all CO/.... and SUB/.... relation words are replaced with COORDINATION and SUBORDINATION)
 | | |-test_reduced_subord_IP.txt (Test file with reduced sentences where all relation words are replaced with COORDINATION)
 | |-input (Use files ending with IP when running the code)
 | | |-sentences.txt (Sentences to be added in training dataset)
 | | |-sentences_more.txt (More sentences to be added in training dataset)
 | | |-train.txt (Training file in raw format)
 | | |-train_IP.txt (Training file in correct format)
 | | |-train_cosubord_IP.txt (Training file with sentences from train_IP.txt where all CO/.... and SUB/.... relation words are replaced with COORDINATION and SUBORDINATION)
 | | |-train_subord_IP.txt (Training file with sentences from train_IP.txt where all relation words are replaced with COORDINATION)
 | | |-train_Level0+1_IP.txt (Training file with all level 0 and half of level 1 sentences in correct format)
 | | |-train_Level0+2+3+4_IP.txt (Training file with all level 4,3,2 and half of level 1 sentences in correct format)
 | | |-train_new.txt (Expanded training file with data from sentences from sentences_more.txt added to train.txt in raw format)
 | | |-train_new_IP.txt (Expanded training file with data from sentences from sentences_more.txt added to train.txt in correct format)
 | |-level_files (Sentences split into level-wise files)
 | | |-graphene.coords
 | | |-Graphene_Level<level_no>.coords (File with level <level_no> sentences in raw format)
 | | |-Graphene_Level<level_no>_IP.coord (File with level <level_no> sentences in correct format)
 | |-output
 | | |-compare_files (.conj like files for manual comparison)
 | | | |-Predictions_<model_name>_<model_size>_b<batch_size>_rel.txt
 | | | |-test_reduced_IP_rel.txt (Gold sentences linearized discourse trees converted into list of sentences.)
 | | |-evaluations
 | | | |-relation_scores
 | | | | |-Result_<model_name>_<model_size>_b<batch_size>_rel.csv (Relation-wise overlap scores)
 | | | | |-sub_stat.csv (Data about relation distribution in Training and Test files)
 | | | |-wire57_f1
 | | | | |-Result_<model_name>_<model_size>_b<batch_size>_wire57v2.txt (Overlap scores for generative models)
 | | | | |-Result_graphene_wire57v2.txt (Overlap scores for Graphene)
 | | |-predictions
 | | | |-Predictions_<model_name>_<model_size>_bs<batch_size>.coords (Predictions of genreative model)
 | | | |-Predictions_graphene.coords (Predictions of Graphene)
 | |-outputNew (Outputs of expanded training dataset. Training was done on train_new_IP.txt)
 | | |-evaluations
 | | | |-wire57_f1
 | | | | |-Result_<model_name>_<model_size>_b<batch_size>_new_v<exp_no>_wire57v2.txt
 | | |-predictions
 | | | |-Predictions_<model_name>_<model_size>_b<batch_size>_new_v<exp_no>.txt
 | |-outputv2
 | | |-compare_files
 | | | |-Predictions_<model_name>_<model_size>_b<batch_size>_v<exp_no>_rel.txt
 | | | |-Predictions_<model_name>_<model_size>_b<batch_size>_v<exp_no>_rel_org.txt
 | | |-evaluations
 | | | |-wire57_f1
 | | | | |-Result_<model_name>_<model_size>_b<batch_size>_v<exp_no>_wire57v2.txt
 | | | | |-Result_T5_base_b03_cosubord_wire57v2.txt (Replaced Co/.... and Sub/.... with COORDINATION and SUBORDINATION)
 | | | | |-Result_T5_base_b03_subord_v1_wire57v2.txt (Replaced everything with SUBORDINATION)
 | | | | |-Result_T5_base_b03_PoS_wire57v2.txt (Parts-of-Speech Tagging experiment)
 | | | | |-Result_T5_base_b03_PoSsubord_v1.txt (Parts-of-Speech Tagging with SUBORDINATION experiment)
 | | | | |-Result_T5_small_b03_CL_wire57v2.txt (Custom Loss Experiment)
 | | | | |-Result_T5_small_b03_ed_e_wire57v2.txt (Ablation Study : Encoder parameters frozen)
 | | | | |-Result_T5_small_b03_ed_d_wire57v2.txt (Ablation Study : Decoder parameters frozen)
 | | |-predictions
 | | | |-LIDC_Predictions_T5_base_b03.txt
 | | | |-Predictions_<model_name>_<model_size>_b<batch_size>_v<exp_no>.txt
 | | | |-Predictions_T5_base_b03_cosubord.txt (Replaced Co/.... and Sub/.... with COORDINATION and SUBORDINATION)
 | | | |-Predictions_T5_base_b03_subord_v1.txt (Replaced everything with SUBORDINATION)
 | | | |-Predictions_T5_base_b03_PoS.txt (Parts-of-Speech Tagging experiment)
 | | | |-Predictions_T5_base_b03_PoSsubord_v1.txt (Parts-of-Speech Tagging with SUBORDINATION experiment)
 | | | |-Predictions_T5_small_b03_CL.txt (Custom Loss Experiment)
 | | | |-Predictions_T5_small_b03_ed_e.txt (Ablation Study : Encoder parameters frozen)
 | | | |-Predictions_T5_small_b03_ed_d.txt (Ablation Study : Decoder parameters frozen)
```

## Models
Trained models can be found in Hugging Face for subordination Task
##### `T5 Base`: https://huggingface.co/bphclegalie/t5-base-legen <br>
##### `T5 Small`: https://huggingface.co/bphclegalie/t5-small-custom-loss


## Prompts

#### GPT models are trained with 3 sets of prompts for the subordination task:
##### 1) Restricted Prompts: Similar to zer shot but the model was provided with 10 sets of relations.
##### 2) Unrestricted Relation: The model has to decipher the relation between the clauses.
##### 3) Few shot Learning: 11 examples were given to model to construct the discourse tree. 

##### The coordination task does not have any `restricted ` and `unrestricted` prompts as it has only one kind of relation. 
