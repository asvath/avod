
# Evaluation of the Aggrate View Object Detection (AVOD) on the Canadian Adverse Driving Conditions Dataset(CADCD) (+ tips to run evaluation on your own dataset)

This respository contains scripts that enables the evaluation of the Aggrate View Object Detection (AVOD) on the Canadian Adverse Driving Conditions Dataset (CADCD). It is not recommended to train AVOD using this respository. The public release of AVOD can be acquired at : https://github.com/kujason/avod 

# Training of AVOD
Training of AVOD was done on the KITTI dataset. Please refer to : https://github.com/kujason/avod 
we trained AVOD to detect cars on the KITTI training set. The AVOD model was trained for 120,000 iterations and was then evaluated on the validation set. The evaluation metrics, 3D AP and AHS at 0.7 IoU were calculated for every 1,000 checkpoints.
We obtained the best scores at checkpoint 83,000. Table 1. shows our evaluationresults (Ours) for the ‘Easy’, ‘Moderate’ and ‘Hard’ car categories at checkpoint 83,000 and the results obtained in: https://arxiv.org/abs/1712.02294 (Standard). Both the results are comparable. Checkpoint 83,000 was thus selected for all our analysis of the performance of AVOD on the CADCD.

TABLE 1

# Evaluation of AVOD

## Getting Started

Recommended folder structure: 

### Folder structure
To utilize the files in this repo, we recommend the following folder structure:
	
	moosey/
	└── training/
		└──annotation
		│   └──0000000[0-7]00.txt
		│
		└──calibmoose
		│   	└── F.yaml, B.yaml etc
	    	│
	    	└──image/
		│   └── 0000000[0-7]00.png
	    	│       
	    	└──lidar_points/
	    	│	└── data/
		│	    └── 0000000001.png
		└── lidar_points/
		│	└── data/
		│	│	└── 0000000000.bin
	     	│	└── timestamps.txt
    		│			
	    	└── planes
            		└──0000000[0-7]00.txt
	

### Mini-batch Generation
The training data needs to be pre-processed to generate mini-batches for the RPN. To configure the mini-batches, you can modify `avod/configs/mb_preprocessing/rpn_[class].config`. You also need to select the *class* you want to train on. Inside the `scripts/preprocessing/gen_mini_batches.py` select the classes to process. By default it processes the *Car* and *People* classes, where the flag `process_[class]` is set to True. The People class includes both Pedestrian and Cyclists. You can also generate mini-batches for a single class such as *Pedestrian* only.

Note: This script does parallel processing with `num_[class]_children` processes for faster processing. This can also be disabled inside the script by setting `in_parallel` to `False`.

```bash
cd avod
python scripts/preprocessing/gen_mini_batches.py
```

Once this script is done, you should now have the following folders inside `avod/data`:
```
data
    label_clusters
    mini_batches
```

### Training Configuration
There are sample configuration files for training inside `avod/configs`. You can train on the example configs, or modify an existing configuration. To train a new configuration, copy a config, e.g. `pyramid_cars_with_aug_example.config`, rename this file to a unique experiment name and make sure the file name matches the `checkpoint_name: 'pyramid_cars_with_aug_example'` entry inside your config.

### Run Trainer
To start training, run the following:
```bash
python avod/experiments/run_training.py --pipeline_config=avod/configs/pyramid_cars_with_aug_example.config
```
(Optional) Training defaults to using GPU device 1, and the `train` split. You can specify using the GPU device and data split as follows:
```bash
python avod/experiments/run_training.py --pipeline_config=avod/configs/pyramid_cars_with_aug_example.config  --device='0' --data_split='train'
```
Depending on your setup, training should take approximately 16 hours with a Titan Xp, and 20 hours with a GTX 1080. If the process was interrupted, training (or evaluation) will continue from the last saved checkpoint if it exists.

### Run Evaluator
To start evaluation, run the following:
```bash
python avod/experiments/run_evaluation.py --pipeline_config=avod/configs/pyramid_cars_with_aug_example.config
```
(Optional) With additional options:
```bash
python avod/experiments/run_evaluation.py --pipeline_config=avod/configs/pyramid_cars_with_aug_example.config --device='0' --data_split='val'
```

The evaluator has two main modes, you can either evaluate a single checkpoint, a list of indices of checkpoints, or repeatedly. The evaluator is designed to be run in parallel with the trainer on the same GPU, to repeatedly evaluate checkpoints. This can be configured inside the same config file (look for `eval_config` entry).

To view the TensorBoard summaries:
```bash
cd avod/data/outputs/pyramid_cars_with_aug_example
tensorboard --logdir logs
```

Note: In addition to evaluating the loss, calculating accuracies, etc, the evaluator also runs the KITTI native evaluation code on each checkpoint. Predictions are converted to KITTI format and the AP is calculated for every checkpoint. The results are saved inside `scripts/offline_eval/results/pyramid_cars_with_aug_example_results_0.1.txt` where `0.1` is the score threshold. IoUs are set to (0.7, 0.5, 0.5) 

### Run Inference
To run inference on the `val` split, run the following script:
```bash
python avod/experiments/run_inference.py --checkpoint_name='pyramid_cars_with_aug_example' --data_split='val' --ckpt_indices=120 --device='1'
```
The `ckpt_indices` here indicates the indices of the checkpoint in the list. If the `checkpoint_interval` inside your config is `1000`, to evaluate checkpoints `116000` and `120000`, the indices should be `--ckpt_indices=116 120`. You can also just set this to `-1` to evaluate the last checkpoint.

### Viewing Results
All results should be saved in `avod/data/outputs`. Here you should see `proposals_and_scores` and `final_predictions_and_scores` results. To visualize these results, you can run `demos/show_predictions_2d.py`. The script needs to be configured to your specific experiments. The `scripts/offline_eval/plot_ap.py` will plot the AP vs. step, and print the 5 highest performing checkpoints for each evaluation metric at the moderate difficulty.

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
