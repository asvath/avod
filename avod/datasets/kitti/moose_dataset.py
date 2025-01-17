"""Dataset utils for preparing data for the network."""
import sys
#print(sys.path)

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import itertools
import fnmatch
import os

import numpy as np
import cv2

from wavedata.tools.core import moose_load_calibration
from wavedata.tools.obj_detection import obj_utils

from avod.core import box_3d_encoder
from avod.core import constants
from avod.datasets.kitti import kitti_aug
from avod.datasets.kitti.kitti_utils import KittiUtils



class Sample:
    def __init__(self, name, augs):
        self.name = name
        self.augs = augs


class MooseDataset:
    def __init__(self, dataset_config):


        """
        Initializes directories, and loads the sample list

        Args:
            dataset_config: KittiDatasetConfig
                name: unique name for the dataset
                data_split: "train", "val", "test", "trainval"
                data_split_dir: must be specified for custom "training"
                dataset_dir: Kitti dataset dir if not in default location
                classes: relevant classes
                num_clusters: number of k-means clusters to separate for
                    each class
        """
        # Parse config
        self.config = dataset_config

        self.name = self.config.name
        self.data_split = self.config.data_split
        #self.dataset_dir = os.path.expanduser(self.config.dataset_dir)
        self.dataset_dir = '/media/wavelab/d3cd89ab-7705-4996-94f3-01da25ba8f50/moosey'

        data_split_dir = self.config.data_split_dir

        self.has_labels = self.config.has_labels
        self.cluster_split = self.config.cluster_split

        self.classes = list(self.config.classes)
        self.num_classes = len(self.classes)
        self.num_clusters = np.asarray(self.config.num_clusters)

        self.bev_source = self.config.bev_source
        self.aug_list = self.config.aug_list

        # Determines the network mode. This is initialized to 'train' but
        # is overwritten inside the model based on the mode.
        self.train_val_test = 'train'
        # Determines if training includes all samples, including the ones
        # without anchor_info. This is initialized to False, but is overwritten
        # via the config inside the model.
        self.train_on_all_samples = False

        self._set_up_classes_name()

        # Check that paths and split are valid
        self._check_dataset_dir()

        # Get all files and folders in dataset directory
        all_files = os.listdir(self.dataset_dir)

        # Get possible data splits from txt files in dataset folder
        possible_splits = []
        for file_name in all_files:
            if fnmatch.fnmatch(file_name, '*.txt'):
                possible_splits.append(os.path.splitext(file_name)[0])
        # This directory contains a readme.txt file, remove it from the list
        if 'readme' in possible_splits:
            possible_splits.remove('readme')


        if self.data_split not in possible_splits:
            print("This is the data_split")
            print(self.data_split)
            print("This is the possible splits")
            print(possible_splits)
            raise ValueError("Invalid data split: {}, possible_splits: {}"
                             .format(self.data_split, possible_splits))

        # Check data_split_dir
        # Get possible data split dirs from folder names in dataset folder
        possible_split_dirs = []
        for folder_name in all_files:
            if os.path.isdir(self.dataset_dir + '/' + folder_name):
                possible_split_dirs.append(folder_name)
        if data_split_dir in possible_split_dirs:
            split_dir = self.dataset_dir + '/' + data_split_dir
            self._data_split_dir = split_dir
        else:
            raise ValueError(
                "Invalid data split dir: {}, possible dirs".format(
                    data_split_dir, possible_split_dirs))

        # Batch pointers
        self._index_in_epoch = 0
        self.epochs_completed = 0

        #self._cam_idx =00 #moose's front camera

        # Initialize the sample list
        loaded_sample_names = self.load_sample_names(self.data_split)

        # Augment the sample list
        aug_sample_list = []

        # Loop through augmentation lengths e.g.
        # 0: []
        # 1: ['flip'], ['pca_jitter']
        # 2: ['flip', 'pca_jitter']
        for aug_idx in range(len(self.aug_list) + 1):
            # Get all combinations
            augmentations = list(itertools.combinations(self.aug_list,
                                                        aug_idx))
            for augmentation in augmentations:
                for sample_name in loaded_sample_names:
                    aug_sample_list.append(Sample(sample_name, augmentation))

        self.sample_list = np.asarray(aug_sample_list)
        self.num_samples = len(self.sample_list)

        self._set_up_directories()


        # Setup utils object
        self.kitti_utils = KittiUtils(self)

    # Paths
    @property
    def rgb_image_dir(self):
        return self.image_dir

    @property
    def sample_names(self):
        # This is a property since the sample list gets shuffled for training
        return np.asarray(
            [sample.name for sample in self.sample_list])

    @property
    def bev_image_dir(self):
        raise NotImplementedError("BEV images not saved to disk yet!")

    def _check_dataset_dir(self):
        """Checks that dataset directory exists in the file system

        Raises:
            FileNotFoundError: if the dataset folder is missing
        """
        # Check that dataset path is valid
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError('Dataset path does not exist: {}'
                                    .format(self.dataset_dir))

    def _set_up_directories(self):
        """Sets up data directories."""
        # Setup Directories
        self.image_dir = self._data_split_dir + '/image' #+ str(self._cam_idx)
        self.calib_dir = self._data_split_dir + '/calibmoose' #think about this
        #self.disp_dir = self._data_split_dir + '/disparity'
        self.planes_dir = self._data_split_dir + '/filtered_planes'
        self.velo_dir = self._data_split_dir + '/filtered_lidar_points/data/filtered_'
        self.depth_dir = '/media/wavelab/d3cd89ab-7705-4996-94f3-01da25ba8f50/avod-master/outputs/obj/filtered_3_snow_depth_moose_multiscale' #+ str(self._cam_idx)


        # Labels are always in the training folder
        self.label_dir = self._data_split_dir+ \
            '/annotation'
        #+ str(self._cam_idx)

    def _set_up_classes_name(self):
        # Unique identifier for multiple classes
        if self.num_classes > 1:
            if self.classes == ['Pedestrian', 'Cyclist']:
                self.classes_name = 'People'
            elif self.classes == ['Car', 'Pedestrian', 'Cyclist']:
                self.classes_name = 'All'
            else:
                raise NotImplementedError('Need new unique identifier for '
                                          'multiple classes')
        else:
            self.classes_name = self.classes[0]

    # Get sample paths
    def get_rgb_image_path(self, sample_name):
        return self.rgb_image_dir + '/' + sample_name + '.png'

    def get_depth_map_path(self, sample_name):
        return self.depth_dir + '/' + sample_name + '_left_depth.png' #what's this

    def get_velodyne_path(self, sample_name):
        return self.velo_dir + '/' + sample_name + '.bin'

    def get_bev_sample_path(self, sample_name):
        return self.bev_image_dir + '/' + sample_name + '.png' #what's this

    # Cluster info
    def get_cluster_info(self):
        return self.kitti_utils.clusters, self.kitti_utils.std_devs

    def get_anchors_info(self, sample_name):
        return self.kitti_utils.get_anchors_info(
            self.classes_name,
            self.kitti_utils.anchor_strides,
            sample_name)

    # Data loading methods
    def load_sample_names(self, data_split):
        """Load the sample names listed in this dataset's set file
        (e.g. train.txt, validation.txt)

        Args:
            data_split: override the sample list to load (e.g. for clustering)

        Returns:
            A list of sample names (file names) read from
            the .txt file corresponding to the data split
        """
        set_file = self.dataset_dir + '/' + data_split + '.txt'
        with open(set_file, 'r') as f:
            sample_names = f.read().splitlines() #eg 00000000001
            print(sample_names)
            print(set_file)

        return np.array(sample_names)

    def load_samples(self, indices):
        """ Loads input-output data for a set of samples. Should only be
            called when a particular sample dict is required. Otherwise,
            samples should be provided by the next_batch function

        Args:
            indices: A list of sample indices from the dataset.sample_list
                to be loaded

        Return:
            samples: a list of data sample dicts
        """
        sample_dicts = []
        for sample_idx in indices:
            sample = self.sample_list[sample_idx]
            sample_name = sample.name

            # Only read labels if they exist
            if self.has_labels:
                # Read mini batch first to see if it is empty
                anchors_info = self.get_anchors_info(sample_name)

                if (not anchors_info) and self.train_val_test == 'train' \
                        and (not self.train_on_all_samples):
                    empty_sample_dict = {
                        constants.KEY_SAMPLE_NAME: sample_name,
                        constants.KEY_ANCHORS_INFO: anchors_info
                    }
                    return [empty_sample_dict]

                obj_labels = obj_utils.read_labels(self.label_dir,
                                                   int(sample_name))

                # Only use objects that match dataset classes
                obj_labels = self.kitti_utils.filter_labels(obj_labels)

            else:
                obj_labels = None

                anchors_info = []

                label_anchors = np.zeros((1, 6))
                label_boxes_3d = np.zeros((1, 7))
                label_classes = np.zeros(1)

            img_idx = int(sample_name)

            # Load image (BGR -> RGB)
            cv_bgr_image = cv2.imread(self.get_rgb_image_path(
                sample_name))
            rgb_image = cv_bgr_image[..., :: -1]
            image_shape = rgb_image.shape[0:2]
            image_input = rgb_image

            # Get ground plane
            ground_plane = obj_utils.get_road_plane(int(sample_name),
                                                    self.planes_dir)



            # Get calibration
            calib = moose_load_calibration.load_calibration(self.calib_dir)

            if img_idx < 100:



                T_IMG_CAM = np.eye(4);  # identity matrix
                T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM00']['camera_matrix']['data']).reshape(-1,3) # camera to image #intrinsic matrix

                # T_IMG_CAM : 4 x 4 matrix
                T_IMG_CAM = T_IMG_CAM[0:3, 0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

                stereo_calib_p2 = T_IMG_CAM

            elif (img_idx >= 100 and img_idx <200):
                T_IMG_CAM = np.eye(4);  # identity matrix
                T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM01']['camera_matrix']['data']).reshape(-1,
                                                                                                3)  # camera to image #intrinsic matrix

                # T_IMG_CAM : 4 x 4 matrix
                T_IMG_CAM = T_IMG_CAM[0:3,0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

                stereo_calib_p2 = T_IMG_CAM

            elif (img_idx >= 200 and img_idx <300):
                T_IMG_CAM = np.eye(4);  # identity matrix
                T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM02']['camera_matrix']['data']).reshape(-1,
                                                                                                3)  # camera to image #intrinsic matrix

                # T_IMG_CAM : 4 x 4 matrix
                T_IMG_CAM = T_IMG_CAM[0:3,0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

                stereo_calib_p2 = T_IMG_CAM

            elif (img_idx >= 300 and img_idx <400):
                T_IMG_CAM = np.eye(4);  # identity matrix
                T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM03']['camera_matrix']['data']).reshape(-1,
                                                                                                3)  # camera to image #intrinsic matrix

                # T_IMG_CAM : 4 x 4 matrix
                T_IMG_CAM = T_IMG_CAM[0:3,0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

                stereo_calib_p2 = T_IMG_CAM

            elif (img_idx >= 400 and img_idx <500):
                T_IMG_CAM = np.eye(4);  # identity matrix
                T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM04']['camera_matrix']['data']).reshape(-1,
                                                                                                3)  # camera to image #intrinsic matrix

                # T_IMG_CAM : 4 x 4 matrix
                T_IMG_CAM = T_IMG_CAM[0:3,0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

                stereo_calib_p2 = T_IMG_CAM

            elif (img_idx >= 500 and img_idx <600):
                T_IMG_CAM = np.eye(4);  # identity matrix
                T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM05']['camera_matrix']['data']).reshape(-1,
                                                                                                3)  # camera to image #intrinsic matrix

                # T_IMG_CAM : 4 x 4 matrix
                T_IMG_CAM = T_IMG_CAM[0:3,0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

                stereo_calib_p2 = T_IMG_CAM

            elif (img_idx >= 600 and img_idx <700):
                T_IMG_CAM = np.eye(4);  # identity matrix
                T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM06']['camera_matrix']['data']).reshape(-1,
                                                                                                3)  # camera to image #intrinsic matrix

                # T_IMG_CAM : 4 x 4 matrix
                T_IMG_CAM = T_IMG_CAM[0:3,0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

                stereo_calib_p2 = T_IMG_CAM

            elif (img_idx >= 700 and img_idx <800):
                T_IMG_CAM = np.eye(4);  # identity matrix
                T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM07']['camera_matrix']['data']).reshape(-1,
                                                                                                3)  # camera to image #intrinsic matrix

                # T_IMG_CAM : 4 x 4 matrix
                T_IMG_CAM = T_IMG_CAM[0:3,0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

                stereo_calib_p2 = T_IMG_CAM

            else:
                print("YOLO")



            point_cloud = self.get_point_cloud('lidar', img_idx, image_shape) #possibly change this

            # Augmentation (Flipping)
            if kitti_aug.AUG_FLIPPING in sample.augs:
                image_input = kitti_aug.flip_image(image_input)
                point_cloud = kitti_aug.flip_point_cloud(point_cloud)
                obj_labels = [kitti_aug.flip_label_in_3d_only(obj)
                              for obj in obj_labels]
                ground_plane = kitti_aug.flip_ground_plane(ground_plane)
                stereo_calib_p2 = kitti_aug.flip_stereo_calib_p2(
                    stereo_calib_p2, image_shape)

            # Augmentation (Image Jitter)
            if kitti_aug.AUG_PCA_JITTER in sample.augs:
                image_input[:, :, 0:3] = kitti_aug.apply_pca_jitter(
                    image_input[:, :, 0:3])

            if obj_labels is not None:
                label_boxes_3d = np.asarray(
                    [box_3d_encoder.object_label_to_box_3d(obj_label)
                     for obj_label in obj_labels])

                label_classes = [
                    self.kitti_utils.class_str_to_index(obj_label.type)
                    for obj_label in obj_labels]
                label_classes = np.asarray(label_classes, dtype=np.int32)

                # Return empty anchors_info if no ground truth after filtering
                if len(label_boxes_3d) == 0:
                    anchors_info = []
                    if self.train_on_all_samples:
                        # If training without any positive labels, we cannot
                        # set these to zeros, because later on the offset calc
                        # uses log on these anchors. So setting any arbitrary
                        # number here that does not break the offset calculation
                        # should work, since the negative samples won't be
                        # regressed in any case.
                        dummy_anchors = [[-1000, -1000, -1000, 1, 1, 1]]
                        label_anchors = np.asarray(dummy_anchors)
                        dummy_boxes = [[-1000, -1000, -1000, 1, 1, 1, 0]]
                        label_boxes_3d = np.asarray(dummy_boxes)
                    else:
                        label_anchors = np.zeros((1, 6))
                        label_boxes_3d = np.zeros((1, 7))
                    label_classes = np.zeros(1)
                else:
                    label_anchors = box_3d_encoder.box_3d_to_anchor(
                        label_boxes_3d, ortho_rotate=True)

            # Create BEV maps
            bev_images = self.kitti_utils.create_bev_maps(
                point_cloud, ground_plane)

            height_maps = bev_images.get('height_maps')
            density_map = bev_images.get('density_map')
            bev_input = np.dstack((*height_maps, density_map))

            sample_dict = {
                constants.KEY_LABEL_BOXES_3D: label_boxes_3d,
                constants.KEY_LABEL_ANCHORS: label_anchors,
                constants.KEY_LABEL_CLASSES: label_classes,

                constants.KEY_IMAGE_INPUT: image_input,
                constants.KEY_BEV_INPUT: bev_input,

                constants.KEY_ANCHORS_INFO: anchors_info,

                constants.KEY_POINT_CLOUD: point_cloud,
                constants.KEY_GROUND_PLANE: ground_plane,
                constants.KEY_STEREO_CALIB_P2: stereo_calib_p2,

                constants.KEY_SAMPLE_NAME: sample_name,
                constants.KEY_SAMPLE_AUGS: sample.augs
            }
            sample_dicts.append(sample_dict)

        return sample_dicts

    def _shuffle_samples(self):
        perm = np.arange(self.num_samples)
        np.random.shuffle(perm)
        self.sample_list = self.sample_list[perm]

    def next_batch(self, batch_size, shuffle=True):
        """
        Retrieve the next `batch_size` samples from this data set.

        Args:
            batch_size: number of samples in the batch
            shuffle: whether to shuffle the indices after an epoch is completed

        Returns:
            list of dictionaries containing sample information
        """

        # Create empty set of samples
        samples_in_batch = []

        start = self._index_in_epoch
        # Shuffle only for the first epoch
        if self.epochs_completed == 0 and start == 0 and shuffle:
            self._shuffle_samples()

        # Go to the next epoch
        if start + batch_size >= self.num_samples:

            # Finished epoch
            self.epochs_completed += 1

            # Get the rest examples in this epoch
            rest_num_examples = self.num_samples - start

            # Append those samples to the current batch
            samples_in_batch.extend(
                self.load_samples(np.arange(start, self.num_samples)))

            # Shuffle the data
            if shuffle:
                self._shuffle_samples()

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            # Append the rest of the batch
            samples_in_batch.extend(self.load_samples(np.arange(start, end)))

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            # Append the samples in the range to the batch
            samples_in_batch.extend(self.load_samples(np.arange(start, end)))

        return samples_in_batch






    def get_point_cloud(self, source, img_idx, image_shape=None):

        calib = moose_load_calibration.load_calibration(self.calib_dir)

        def lidar_to_camera(lidar_path, img_idx):  # lidar to front camera
            scan_data = np.fromfile(lidar_path, dtype=np.float32)  # numpy from file reads binary file
            lidar = scan_data.reshape((-1, 4))

            [rows,cols] = lidar.shape;  # rows and cols of the lidar points, rows = number of lidar points, columns (4 = x , y, z , intensity)

            # 0: front , 1: front right, 2: right front, 3: back right, 4: back, 5: left back, 6: left front, 7: front left

            point_cloud_cam = np.zeros([lidar.shape[0], lidar.shape[1]])

            if img_idx < 100:
                T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM00']));


            elif (img_idx >= 100 and img_idx < 200):
                T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM01']));


            elif (img_idx >= 200 and img_idx < 300):
                T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM02']));

            elif (img_idx >= 300 and img_idx < 400):
                T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM03']));

            elif (img_idx >= 400 and img_idx < 500):
                T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM04']));

            elif (img_idx >= 500 and img_idx < 600):
                T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM05']));

            elif (img_idx >= 600 and img_idx < 700):
                T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM06']));

            elif (img_idx >= 700 and img_idx < 800):
                T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM07']));

            else:
                T_CAM_LIDAR = "YOLO"


            for i in range(rows):
                point_cloud_cam[i] = np.matmul(T_CAM_LIDAR, lidar[i])

            return point_cloud_cam


        if source == "lidar":
            # wavedata wants im_size in (w, h) order
            min_intensity = None

            im_size = [image_shape[1], image_shape[0]]

            if img_idx < 100:
                lidar_idx = img_idx
                lidar_path = self.velo_dir  + format(lidar_idx, '010') + ".bin"


            elif (img_idx >= 100 and img_idx < 200):
                lidar_idx = img_idx - 100
                lidar_path = self.velo_dir  + format(lidar_idx, '010') + ".bin"


            elif (img_idx >= 200 and img_idx < 300):
                lidar_idx = img_idx - 200
                lidar_path = self.velo_dir +  format(lidar_idx, '010') + ".bin"

            elif (img_idx >= 300 and img_idx < 400):
                lidar_idx = img_idx - 300
                lidar_path = self.velo_dir +  format(lidar_idx, '010') + ".bin"

            elif (img_idx >= 400 and img_idx < 500):
                lidar_idx = img_idx - 400
                lidar_path = self.velo_dir +   format(lidar_idx, '010') + ".bin"

            elif (img_idx >= 500 and img_idx < 600):
                lidar_idx = img_idx - 500
                lidar_path = self.velo_dir +  format(lidar_idx, '010') + ".bin"

            elif (img_idx >= 600 and img_idx < 700):
                lidar_idx = img_idx - 600
                lidar_path = self.velo_dir +  format(lidar_idx, '010') + ".bin"

            elif (img_idx >= 700 and img_idx < 800):
                lidar_idx = img_idx - 700
                lidar_path = self.velo_dir +  format(lidar_idx, '010') + ".bin"
            else:
                lidar_path = "YOLO"




            pts = lidar_to_camera(lidar_path,img_idx)

            # The given image is assumed to be a 2D image
            if not im_size:

                return pts[:, 0:3].T





            else:


                # Only keep points in front of camera (positive z)
                pts = pts[pts[:, 2] > 0]

                if img_idx < 100:

                    T_IMG_CAM = np.eye(4);  # identity matrix
                    T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM00']['camera_matrix']['data']).reshape(-1,3)  # camera to image #intrinsic matrix

                    # T_IMG_CAM : 4 x 4 matrix
                    T_IMG_CAM = T_IMG_CAM[0:3,0:4];  # remove last row, #choose the first 3 rows and get rid of the last column



                elif (img_idx >= 100 and img_idx < 200):
                    T_IMG_CAM = np.eye(4);  # identity matrix
                    T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM01']['camera_matrix']['data']).reshape(-1,3)  # camera to image #intrinsic matrix

                    # T_IMG_CAM : 4 x 4 matrix
                    T_IMG_CAM = T_IMG_CAM[0:3,0:4];  # remove last row, #choose the first 3 rows and get rid of the last column



                elif (img_idx >= 200 and img_idx < 300):
                    T_IMG_CAM = np.eye(4);  # identity matrix
                    T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM02']['camera_matrix']['data']).reshape(-1,3)  # camera to image #intrinsic matrix

                    # T_IMG_CAM : 4 x 4 matrix
                    T_IMG_CAM = T_IMG_CAM[0:3,0:4];  # remove last row, #choose the first 3 rows and get rid of the last column



                elif (img_idx >= 300 and img_idx < 400):
                    T_IMG_CAM = np.eye(4);  # identity matrix
                    T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM03']['camera_matrix']['data']).reshape(-1,3)  # camera to image #intrinsic matrix

                    # T_IMG_CAM : 4 x 4 matrix
                    T_IMG_CAM = T_IMG_CAM[0:3,0:4];  # remove last row, #choose the first 3 rows and get rid of the last column



                elif (img_idx >= 400 and img_idx < 500):
                    T_IMG_CAM = np.eye(4);  # identity matrix
                    T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM04']['camera_matrix']['data']).reshape(-1,3)  # camera to image #intrinsic matrix

                    # T_IMG_CAM : 4 x 4 matrix
                    T_IMG_CAM = T_IMG_CAM[0:3,0:4];  # remove last row, #choose the first 3 rows and get rid of the last column



                elif (img_idx >= 500 and img_idx < 600):
                    T_IMG_CAM = np.eye(4);  # identity matrix
                    T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM05']['camera_matrix']['data']).reshape(-1,3)  # camera to image #intrinsic matrix

                    # T_IMG_CAM : 4 x 4 matrix
                    T_IMG_CAM = T_IMG_CAM[0:3,0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

                elif (img_idx >= 600 and img_idx < 700):
                    T_IMG_CAM = np.eye(4);  # identity matrix
                    T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM06']['camera_matrix']['data']).reshape(-1,3)  # camera to image #intrinsic matrix

                    # T_IMG_CAM : 4 x 4 matrix
                    T_IMG_CAM = T_IMG_CAM[0:3,0:4];  # remove last row, #choose the first 3 rows and get rid of the last column


                elif (img_idx >= 700 and img_idx < 800):
                    T_IMG_CAM = np.eye(4);  # identity matrix
                    T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM07']['camera_matrix']['data']).reshape(-1,
                                                                                                    3)  # camera to image #intrinsic matrix

                    # T_IMG_CAM : 4 x 4 matrix
                    T_IMG_CAM = T_IMG_CAM[0:3,0:4];  # remove last row, #choose the first 3 rows and get rid of the last column


                else:
                    T_IMG_CAM = "yolo"



                # read image


                point_in_im = np.zeros([pts.shape[0], 2])
                for i in range(len(pts)):


                    a = np.matmul(T_IMG_CAM, pts[i])

                    point_in_im[i] = [a[0] / a[2], a[1] / a[2]]

                # Project points onto image

                # Project to image frame

                # Filter based on the given image size
                image_filter = (point_in_im[:, 0] > 0) & \
                               (point_in_im[:, 0] < im_size[0]) & \
                               (point_in_im[:, 1] > 0) & \
                               (point_in_im[:, 1] < im_size[1])

            if not min_intensity:

                return pts[image_filter][:, 0:3].T

            else:
                intensity_filter = i > min_intensity
                point_filter = np.logical_and(image_filter, intensity_filter)
                return pts[point_filter][:, 0:3].T




        else:
            raise ValueError("Invalid source {}".format(source))

            #img = cv2.imread(self.img_path)

            #cv2.imshow('image', img)
            #image_shape = img.shape[:2]


    '''

    def lidar_to_camera_zero(img_idx):  # lidar to front camera
        lidar_path = self.velo_dir + "/" + format(img_idx, '010') + ".bin"
        scan_data = np.fromfile(lidar_path, dtype=np.float32)  # numpy from file reads binary file
        lidar = scan_data.reshape((-1, 4))

        [rows,
         cols] = lidar.shape;  # rows and cols of the lidar points, rows = number of lidar points, columns (4 = x , y, z , intensity)

        # 0: front , 1: front right, 2: right front, 3: back right, 4: back, 5: left back, 6: left front, 7: front left

        point_cloud_cam = np.zeros([lidar.shape[0], lidar.shape[1]])

        T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM00']));  # going from lidar to camera

        for i in range(rows):
            point_cloud_cam[i] = np.matmul(T_CAM_LIDAR, lidar[i])

        return point_cloud_cam



    def get_lidar_point_cloud(point_cloud_cam,
                              image_shape=None, min_intensity=None):

        im_size = [image_shape[1], image_shape[0]]
        """ Calculates the lidar point cloud, and optionally returns only the
        points that are projected to the image.

        :param img_idx: image index
        :param calib_dir: directory with calibration files
        :param velo_dir: directory with velodyne files
        :param im_size: (optional) 2 x 1 list containing the size of the image
                          to filter the point cloud [w, h]
        :param min_intensity: (optional) minimum intensity required to keep a point

        :return: (3, N) point_cloud in the form [[x,...][y,...][z,...]]
        """

        pts = point_cloud_cam[:, 0:3]

        # The given image is assumed to be a 2D image
        if not im_size:

            return pts.T

        else:
            # Only keep points in front of camera (positive z)
            pts = pts[pts[:, 2] > 0]
            calib = moose_load_calibration.load_calibration(self.calib_dir)
            T_IMG_CAM = np.eye(4);  # identity matrix
            T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM0' + cam]['camera_matrix']['data']).reshape(-1,
                                                                                                 3)  # matrix of the camera
            T_IMG_CAM = T_IMG_CAM[0:3, 0:4];  # remove last row

            T_CAM_LIDAR = np.linalg.inv(np.array(
                calib['extrinsics']['T_LIDAR_CAM0' + cam]))  # lidar to camera is the inverse of camera to lidar

            dist_coeffs = np.array(calib['CAM0' + cam]['distortion_coefficients']['data'])  # get the distortion coeff

            moose_lidar_utils_obj = lidar_utils(T_CAM_LIDAR)  # T_CAM_LIDAR (lidar to camera)

            # read image
            img = cv2.imread(img_path)

            DISTORTED = False

            # Project points onto image
            img, img_points = moose_lidar_utils_obj.project_points(img, lidar_path, T_IMG_CAM, T_CAM_LIDAR, dist_coeffs,
                                                                   DISTORTED,
                                                                   'bin')  # projects lidar points into image frame
            # Project to image frame
            point_in_im = img_points

            # Filter based on the given image size
            image_filter = (point_in_im[:, 0] > 0) & \
                           (point_in_im[:, 0] < im_size[0]) & \
                           (point_in_im[:, 1] > 0) & \
                           (point_in_im[:, 1] < im_size[1])

        if not min_intensity:
            return pts[image_filter].T

        else:
            intensity_filter = i > min_intensity
            point_filter = np.logical_and(image_filter, intensity_filter)
            return pts[point_filter].T
    
    '''