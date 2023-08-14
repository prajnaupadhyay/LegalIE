# LegalIE

Python 3.10 is needed to run the code.

`conda create -n legalIE python=3.10`

`conda activate legalIE`

** #Generating split sentences from the conjunction model**

**Dataset 1**: We add coordination at the beginning of every sentence obtained from the conjunctive model of openIE6.

To obtain split sentence hierarchy, 

python run.py --mode splitpredict --inp sentences.txt --out predictions.txt --rescoring --task oie --gpus 1 --oie_model models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt --conj_model models/conj_model/epoch=28_eval_acc=0.854.ckpt --rescore_model models/rescore_model --num_extractions 5 

predictions.txt.conj obtained from running openIE6 is processed using SplitCoordiantion.py file to obtain split hierarchy.

prediction.txt.conj contents : 

Bell , based in Los Angeles , makes and distributes electronic , computer and building products . <br>
Bell , based in Los Angeles , makes electronic products . <br>
Bell , based in Los Angeles , makes computer products . <br>
Bell , based in Los Angeles , makes building products .<br>
Bell , based in Los Angeles , distributes electronic products . <br>
Bell , based in Los Angeles , distributes computer products . <br>
Bell , based in Los Angeles , distributes building products . <br>

**cmd:** python SplitCoordination.py predictions.txt.conj outputSplit.txt

output:
#Bell , based in Los Angeles , makes and distributes electronic , computer and building products .
COORDINATION( Bell , based in Los Angeles , makes electronic products .,COORDINATION( Bell , based in Los Angeles , makes computer products .,COORDINATION( Bell , based in Los Angeles , makes building products .,COORDINATION( Bell , based in Los Angeles , distributes electronic products .,COORDINATION( Bell , based in Los Angeles , distributes computer products .,COORDINATION( Bell , based in Los Angeles , distributes building products .,))))))

**Dataset 2**: In order to obtain coordiantion sentences at every level, make the changes as shown in data.py file of openIe6 and then run

python run.py --mode splitpredict --inp sentences.txt --out predictions.txt --rescoring --task oie --gpus 1 --oie_model models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt --conj_model models/conj_model/epoch=28_eval_acc=0.854.ckpt --rescore_model models/rescore_model --num_extractions 5 


This generates tree hierarchy of sentences in out file of slurm job, which can be processed using MergeIO.py.
To use MergeIO.py, 
1. Remove special characters and update line starting with " Testing".
2. input sentences.txt and slurm.jobId.out file as input.
3. Add brackets
**cmd:**
1. python MergeIO.py slurm.jobId.out withoutSplChar.txt
2. python input.txt withoutSplChar.txt mergeIO.txt (Merges input and output)
3. Add brackets to mergeIO.txt
4. Add NONE between two input sentences
   
output :
#Bell , based in Los Angeles , makes and distributes electronic , computer and building products .
COORDINATION( ['Bell , based in Los Angeles , makes electronic products .', 'Bell , based in Los Angeles , makes computer products .', 'Bell , based in Los Angeles , makes building products .', 'Bell , based in Los Angeles , distributes electronic products .', 'Bell , based in Los Angeles , distributes computer products .', 'Bell , based in Los Angeles , distributes building products .'] )

**Dataset3** To obtain linearized output from Graphene's tree hierarchical sentence structure, run ProcessingDisSimTree.py.
Input: treeStructure.txt output:LinearizedTree.txt


