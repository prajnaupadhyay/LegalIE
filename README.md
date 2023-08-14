# LegalIE

Python 3.10 is needed to run the code.

`conda create -n legalIE python=3.10`

`conda activate legalIE`

#Generating split sentences from the conjunction model

Dataset 1: We add coordination at the beginning of every sentence obtained from conjunctive model of openIE6.

To obtain split sentence hierarchy, 

python run.py --mode splitpredict --inp sentences.txt --out predictions.txt --rescoring --task oie --gpus 1 --oie_model models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt --conj_model models/conj_model/epoch=28_eval_acc=0.854.ckpt --rescore_model models/rescore_model --num_extractions 5 

predictions.txt.conj obtained from running openIE6 is processed using SplitCoordiantion.py file to obtain split hierarchy.

Dataset 2: In order to obtain coordiantion sentences at every level, make the changes as shown in data.py file of openIe6 and then run

python run.py --mode splitpredict --inp sentences.txt --out predictions.txt --rescoring --task oie --gpus 1 --oie_model models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt --conj_model models/conj_model/epoch=28_eval_acc=0.854.ckpt --rescore_model models/rescore_model --num_extractions 5 

This generates tree hierarchy of sentences in out file of slurm job, which can be processed using MergeIO.py



