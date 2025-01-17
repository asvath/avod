
# Evaluation of the Aggrate View Object Detection (AVOD) on the Canadian Adverse Driving Conditions Dataset(CADCD) (+ tips to run evaluation on your own dataset)

This respository contains scripts that enable the evaluation of the Aggrate View Object Detection (AVOD) on the Canadian Adverse Driving Conditions Dataset (CADCD). It is **not** recommended to train AVOD using this respository as several files were modified for the evaluation on CADCD. To train AVOD, refer to the public release of AVOD which can be acquired at : https://github.com/kujason/avod (we will reference this repository as [1])

**The list of files that were modified from the original AVOD repo [1] or were newly created can be found in the `Evaluation of AVOD on CADCD` section below. Please ensure that you have read the descriptions of the files that were changed/added and make the necessary replacements to your own AVOD repo before proceeding to evaluate AVOD on CADCD or your own dataset. Note: This work can be cleaned up further to ensure that training and evaluation can be performed from the same repo using your own dataset.**



# Training of AVOD
Training of AVOD was done on the KITTI dataset. Please refer to [1]. We trained AVOD to detect cars on the KITTI training set. The AVOD model was trained for 120,000 iterations and was then evaluated on the validation set. The evaluation metrics, 3D AP and AHS at 0.7 IoU were calculated for every 1,000 checkpoints. 
We obtained the best scores at checkpoint 83,000. Our results at checkpoint 83,000 and the results obtained in: https://arxiv.org/abs/1712.02294 (Standard) were comparable. Checkpoint 83,000 was thus selected for all our analysis of the performance of AVOD on the CADCD. Ensure that your training results are comparable to Standard if you are training on KITTI. 

# Evaluation of AVOD on CADCD (+ tips to run evaluation on other datasets)

## Getting Started

**Before proceeding, please ensure that you have read through the list of files that were changed/added. Replace the files in your AVOD repo which you acquired from [1] using the modified files/new files from this repo. Ensure that this is done before proceeding to evaluate AVOD on CADCD. Note that we have provided some descriptions of the changes made to aid anyone who wants to try modifying AVOD for other datasets.**

**IMPORTANT: Any of the changes below that were made on the wavedata folder can be found at https://github.com/asvath/wavedata instead of this repo or [1]**

### List of files that were modified from/added to the original AVOD repo [1]:

#### 1: scripts/preprocessing/gen_mini_batches.py
 Ensure that 'process_ppl = True ' is set to False.
#### 2: avod/builders/dataset_builders.py 
- Change 'from avod.datasets.kitti.kitti_dataset import KittiDataset' to 'from avod.datasets.kitti.moose_dataset import MooseDataset' [see 4] (or whatever you call your dataset class)
- Modify 'KITTI_VAL = KittiDatasetConfig' 
- Ensure that @staticmethod points to your dataset e.g: @staticmethod
    def build_kitti_dataset(base_cfg,use_defaults=True,new_cfg=None) -> MooseDataset
- Ensure that @staticmethod returns your dataset e.g MooseDataset(cfg_copy)
- Change DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_TRAIN_MINI) to DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_VAL)

#### 3: avod/core/evaluator_utils.py
- Remove 'from wavedata.tools.core import calib_utils' (This is important as you do not want to call the wrong calibration)
- Import your own calibration file : e.g from wavedata.tools.core import moose_load_calibration [see 5]
- Ensure that stereo_calib_p2 is your camera_matrix (CAM to IMG)

Note that we refined stereo_calib_p2 in the script itself. It will be better to define p2 in your own calibration file (e.g moose_load_calibration) and import p2 whenever you need it instead of defining it everywhere. We have yet to do that.

#### 4: Create MooseClass : avod/datasets/kitti/moose_dataset.py
This is a new class for the CADCD dataset (create similar file for your own dataset)
- Includes directory setup info 
- Definition of transformation matrix from camera to img frame: stereo_calib_p2
- get_point_cloud function that gets the lidar point cloud

Note: This file was modified using avod/datasets/kitti/kitti_dataset.py

#### 5: Create calibration file for your dataset: wavedata/tools/core/moose_load_calibration.py 

This file was acquired from the CADCD devkit (https://github.com/wavelab/cadcd_devkit : load_calibration.py) 
We call the file moose_load_calibration.py. Make a similar file for other datasets.

#### 6: wavedata/tools/obj_detection/obj_utils
- import wavedata/tools/core/calib_utils.py (note we normally do not import this, see 8.)
- import your own calibration file (e.g wavedata/tools/core/moose_load_calibration.py)
- import wavedata/tools/core/depth_map_utils.py (we create this, see 7., it's mainly for 3D visualization purposes, not needed otherwise)
- change read_labels to suit the way your labels are named e.g (label_dir + "/" +"%010d.txt" % img_idx) instead of 
(label_dir + "/%06d.txt" % img_idx)
- define get_depth_map_point_cloud
- change get_road_planes to load ground planes that you created for your own dataset

#### 7: wavedata/tools/core/depth_map_utils.py
This was from AVOD's development repo (not public). Enjoy. 

#### 8: wavedata/tools/core/calib_utils.py
- project_pc_to_image function was added (from AVOD's development repo (not public). Enjoy.)

Note: We normally do not import calib_utils (as it works for the KITTI dataset and not the CADCD). However, the exception is we import calib_utils in obj_utils.py as we want the get_stereo_calibration and project_to_image functions to create 3D visualizations of our predictions. If you do not want 3D visualizations, this can be ignored.

#### 9: avod/datasets/kitti/kitti_utils.py 
- import your calibration file
- import opencv
- define a new get_point_cloud function (remove/comment out the original)

#### 10: demos/moose_show_predictions_3d.py (making this script this required a lot of tracking down from multiple places like AVOD public repo, Jason's scene vis repo, Jason's ip basic repo (https://github.com/kujason) etc).

We have tried to ensure, that you do not have to track things down all over the place. (Feel free to visit the other repos if you want to see where most of these are coming from)
- import wavedata.tools.visualization import vis_utils (see 11.)

#### 11: wavedata/wavedata/tools/visualization/vis_utils.py

- Ensure that you are reading in your images correctly : edit this line: img = np.array(Image.open("%s/%010d.png" % (image_dir, index)),dtype=np.uint8)
- project_img_to_point function was added
- import vtk
- import moose_load_calibration/your own calibration file for your dataset
- import wavedata.tools.core import calib_utils
- Add a new class ToggleActorsInteractorStyle

#### 12: wavedata/wavedata/tools/visualization 
We have many new files that aren't in the original repo [1]. These come from places such as AVOD's development repo (not open to public), Jason's scene vis repo, Jason's ip basic repo (https://github.com/kujason). We have tried to ensure, that you do not have to track things down all over the place. (Feel free to visit the other repos if you want to see where most of these are coming from)

#### 13: demos/moose_show_predictions_2d.py 
- remove original calibration (for KITTI)
- import wavedata/tools/core/moose_load_calibration.py or your own calibration 
- change 'global_step' to the checkpoint used for evaluation
- change 'checkpoint_name' to reflect the config file that was used
- calib_p2 was defined differently than the orig file

#### 14: scripts/preprocessing/save_depth_maps.py 
 - You need to ensure that Jason's IP basic repo is located in your AVOD repo
 - import ip_basic
 - from ip_basic.ip_basic import ip_depth_map_utils

This is run to create depth maps for 3D visualization purposes
#### 15: Others
- Ensure that scripts/offline_eval/save_kitti_predictions.py have the correct checkpoint name
- When you run the evaluation (See `Run Evaluator` below), you will get ouput files : avod/data/outputs/pyramid_cars_with_aug_example/predictions. Go to predictions/kitti_native_eval. The following files need to be changed:
	- evaluate_object_3d_offline.cpp: `sprintf(file_name,"%010d.txt",indices.at(i));` Ensure that you have the right format to read the ground truth (gt   = loadGroundtruth(gt_dir + "/" + file_name,gt_success);)	
	- evaluate_object_3d_offline.cpp : same changes as above
	- run_eval.sh : Ensure that it is pointing to the correct annotation folder (e.g ./evaluate_object_3d_offline /media/wavelab/d3cd89ab-7705-4996-94f3-01da25ba8f50/moosey/training/annotation/ $2/$3 | tee -a ./$4_results_$2.txt)
	- run_eval_05_iou.sh (same as above)
- Make changes to your config file (see `Evaluation configuration below`)
	



### Recommended folder structure: 

### Folder structure for CADCD
To utilize the files in this repo, we recommend the following folder structure for your dataset:
	
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
	│	└── lidar_points/
	│	│	└── data/
	│	│	│	└── 0000000000.bin
	│     	│	└── timestamps.txt
	│	│			
	│	└── planes
	│		└──0000000[0-7]00.txt
	└──val.txt

Note that we placed all the frames from all the cameras in the same folder (e.g image). The file names were changed to reflect the camera. e.g CAM 7 : original : 0000000008.png new : 0000000708.png. Same goes for annotation files.
There were no changes to the lidar points data as all the frames relate to the same lidar files. Refer to : https://github.com/asvath/cadcd for info on how the annotation files were made. Note that we have included the val.txt file in this repo. It contains the list of files used for evaluation. Files that have no annotations (e.g no objects of interest in the frame) have been excluded from evaluation. Please contact me for the planes if required. More info available at : https://github.com/asvath/cadcd

### Mini-batch Generation
We need to generate mini-batches for the RPN. To configure the mini-batches, you can modify `avod/configs/mb_preprocessing/rpn_cars.config`. Ensure that the *dataset_dir* points to the correct dataset that you want to evaluate on (e.g /home/moosey). Inside the `scripts/preprocessing/gen_mini_batches.py`, notice that the *cars* class is selected for processing (`process_car`) is set to True. 

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

### Evaluation Configuration
We modified `pyramid_cars_with_aug_example.config`, for our evaluation. In particular, we changed the `dataset_dir` to point to our dataset directory. e.g (/home/moosey). `dataset_split: val`. In `eval_config` we set `ckpt_indices :83` and `evaluate_repeatedly: False`. This is due to our checkpoint selection as described in the training of AVOD section above (checkpoint : 83000). 


### Run Evaluator
To start evaluation, run the following:
```bash
python avod/experiments/run_evaluation.py --pipeline_config=avod/configs/pyramid_cars_with_aug_example.config --device='0' --data_split='val'
```
Stop as soon as it starts to run, as some files will be generated and those need to be modified before you can get AP scores for your dataset. Please refer to `List of files that were modified from/added to the original AVOD repo [1]` above. See `.15 Others`.

Note: The results are located at `scripts/offline_eval/results/pyramid_cars_with_aug_example_results_0.1.txt` where `0.1` is the score threshold. IoUs are set to (0.7 for cars) 

Note if you want to revaluate the same checkpoint, go into avod/data/outputs/pyramid_cars_with_aug_example/predictions, and delete the csv files. Run the evaluation again.


### Viewing Results
All results should be saved in `avod/data/outputs`. Here you should see `proposals_and_scores` and `final_predictions_and_scores` results. To visualize these results, you can run `demos/moose_show_predictions_2d.py`, `demos/moose_show_predictions_3d.py` . The images will be saved in avod/data/outputs/pyramid_cars_with_aug_example(checkpoint name)/predictions/images_[2d_3d]
The figure below shows an example of the 3D visualization of our results:


<img src="https://github.com/asvath/avod/blob/master/pics_avod/0000000004.png" width="520" height="400">


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
