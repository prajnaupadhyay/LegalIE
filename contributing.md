## How to run on HPC
1. `cd /scratch/prajna/LegalIE/customloss/`
2. For co-ordination
 - `sbatch job.sh train-test data/CoordinationDataSet/input/train_copy.coord model_dir_T5s_co_03_org_v2 data/CoordinationDataSet/gold/test_copy.coord data/CoordinationDataSet/predictions/<file_name> <model_name> <batch_size> <seed_value>` 
 3. For subordination
 - `batch job.sh train-test data/SubordinationDataSet/input/train_IP.txt model_dir_T5s_sub_03_def data/SubordinationDataSet/gold/test_reduced_IP.txt data/SubordinationDataSet/output/<file_name> <model_name> <batch_size> <seed_value>`
 Notes:
 - `<seed_value>` can be any valid integer
 - `<model_name>` needs to be `T5` and `BART`. Model size actually needs to changed in code.
 - First paths in both commands are paths to train and test data. These wont change in most cases.

## Paths to the needed files (Copy these files for remaining results)
1. `/scratch/prajna/LegalIE/customloss/data/CoordinationDataSet/predictions/Predictions_BART_small_both_v2.txt`
2. `/scratch/prajna/LegalIE/customloss/data/CoordinationDataSet/predictions/Predictions_BART_base_both_v2.txt`
3. `/scratch/prajna/LegalIE/customloss/data/CoordinationDataSet/predictions/Predictions_T5_base_both.tx`


## Commads used for recent files
1. `sbatch job.sh train-test data/CoordinationDataSet/input/train_copy.coord model_dir_Bas_co_32_both data/CoordinationDataSet/gold/test_copy.coord data/CoordinationDataSet/predictions/Predictions_BART_small_both_v2.txt BART 32 1097`
2. `sbatch job.sh train-test data/CoordinationDataSet/input/train_copy.coord model_dir_Bab_co_32_both data/CoordinationDataSet/gold/test_copy.coord data/CoordinationDataSet/predictions/Predictions_BART_base_both_v2.txt BART 32 1097`
3. `sbatch job.sh train-test data/CoordinationDataSet/input/train_copy.coord model_dir_T5b_co_03_both data/CoordinationDataSet/gold/test_copy.coord data/CoordinationDataSet/predictions/Predictions_T5_base_both.txt T5 3 238`