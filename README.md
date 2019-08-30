
# Evaluation of the Aggrate View Object Detection (AVOD) on the Canadian Adverse Driving Conditions Dataset(CADCD) (+ tips to run evaluation on your own dataset)

This respository contains scripts that enables the evaluation of the Aggrate View Object Detection (AVOD) on the Canadian Adverse Driving Conditions Dataset (CADCD). It is not recommended to train AVOD using this respository as several files were modified for the evaluation on CADCD. The public release of AVOD can be acquired at : https://github.com/kujason/avod (we will reference this repository as [1])

The list of files that were changed from the original can be found in Section Evaluation of AVOD on CADCD shown below.





# Training of AVOD
Training of AVOD was done on the KITTI dataset. Please refer to [1]. We trained AVOD to detect cars on the KITTI training set. The AVOD model was trained for 120,000 iterations and was then evaluated on the validation set. The evaluation metrics, 3D AP and AHS at 0.7 IoU were calculated for every 1,000 checkpoints.
We obtained the best scores at checkpoint 83,000. Table 1. shows our evaluationresults (Ours) for the ‘Easy’, ‘Moderate’ and ‘Hard’ car categories at checkpoint 83,000 and the results obtained in: https://arxiv.org/abs/1712.02294 (Standard). Both the results are comparable. Checkpoint 83,000 was thus selected for all our analysis of the performance of AVOD on the CADCD.

TABLE 1

# Evaluation of AVOD on CADCD

## Getting Started

**Before proceeding, please ensure that you have read the list of files that were changed and make the same changes to your AVOD repo.**

Recommended folder structure: 

### Folder structure
To utilize the files in this repo, we recommend the following folder structure:
	
	moosey/
	└── training/
	│	└──annotation
	│	│   └──0000000[0-7]00.txt
	│	│
	│	└──calibmoose
	│	│   	└── F.yaml, B.yaml etc
	│    	│
	│    	└──image/
	│	│   └── 0000000[0-7]00.png
	│    	│       
	│    	└──lidar_points/
	│    	│	└── data/
	│	│	    └── 0000000001.png
	│	└── lidar_points/
	│	│	└── data/
	│	│	│	└── 0000000000.bin
	│     	│	└── timestamps.txt
	│	│			
	│	└── planes
	│		└──0000000[0-7]00.txt
	└──val.txt

### Mini-batch Generation
We need to generate mini-batches for the RPN. To configure the mini-batches, you can modify `avod/configs/mb_preprocessing/rpn_cars.config`. Ensure that your dataset_dir points to the correct dataset that you want to evaluate on (e.g /home/moosey). Inside the `scripts/preprocessing/moose_gen_mini_batches.py`, notice that the *cars* class is selected for processing (`process_car`) is set to True. 

```bash
cd avod
python scripts/preprocessing/moose_gen_mini_batches.py
```

Once this script is done, you should now have the following folders inside `avod/data`:
```
data
    label_clusters
    mini_batches
```

### Evaluation Configuration
We modified `pyramid_cars_with_aug_example.config`, for our evaluation. In particular, we changed the `dataset_dir` to point to our dataset directory. e.g (/home/moosey). `dataset_split: val`. In `eval_config` we set `ckpt_indices :83` and `evaluate_repeatedly: False`. This is due to our checkpoint selection as described in the training of AVOD section above (checkpoint : 83000). 


### Run Evaluator
To start evaluation, run the following:
```bash
python avod/experiments/run_evaluation.py --pipeline_config=avod/configs/pyramid_cars_with_aug_example.config --device='0' --data_split='val'
```

Note: The results are located at `scripts/offline_eval/results/pyramid_cars_with_aug_example_results_0.1.txt` where `0.1` is the score threshold. IoUs are set to (0.7 for cars) 


### Viewing Results
All results should be saved in `avod/data/outputs`. Here you should see `proposals_and_scores` and `final_predictions_and_scores` results. To visualize these results, you can run `demos/moose_show_predictions_2d.py`. 


## LICENSE
Copyright (c) 2018 [Jason Ku](https://github.com/kujason), [Melissa Mozifian](https://github.com/melfm), [Ali Harakeh](www.aharakeh.com), [Steven L. Waslander](http://wavelab.uwaterloo.ca)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
