import os
import sys
import time

import numpy as np

import ip_basic
from ip_basic.ip_basic import ip_depth_map_utils

import avod
from avod.builders.dataset_builder import DatasetBuilder
from wavedata.tools.core import moose_load_calibration
from wavedata.tools.obj_detection import obj_utils
import cv2


#from wavedata.tools.core import calib_utils

from wavedata.tools.core import depth_map_utils


def main():
    """Interpolates the lidar point cloud to and saves a dense depth map of the scene.
    """

    ##############################
    # Options
    ##############################

    dataset = DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_VAL)

    # Fill algorithm ('ip_basic_{...}')
    fill_type = 'multiscale'

    save_depth_maps = True

    out_depth_map_dir = 'outputs/obj/filtered_3_snow_depth_moose_{}'.format(fill_type)
    print(out_depth_map_dir)

    samples_to_use = None
    # samples_to_use = ['000764']

    ##############################
    # End of Options
    ##############################
    os.makedirs(out_depth_map_dir, exist_ok=True)

    # Rolling average array of times for time estimation
    avg_time_arr_length = 5
    last_fill_times = np.repeat([1.0], avg_time_arr_length)
    last_total_times = np.repeat([1.0], avg_time_arr_length)


    if samples_to_use is None:
        samples_to_use = [sample.name for sample in dataset.sample_list]

    for sample_idx, sample_name in enumerate(samples_to_use):

        # Calculate average time with last n fill times
        avg_fill_time = np.mean(last_fill_times)
        avg_total_time = np.mean(last_total_times)

        # Print progress
        sys.stdout.write('\rProcessing {} / {}, Idx {}, Avg Fill Time: {:.5f}s, '
                         'Avg Time: {:.5f}s, Est Time: {:.3f}s'.format(
                             sample_idx, dataset.num_samples - 1, sample_name,
                             avg_fill_time, avg_total_time,
                             avg_total_time * (dataset.num_samples - sample_idx)))
        sys.stdout.flush()

        # Start timing
        start_total_time = time.time()

        # Load sample info
        image = cv2.imread(dataset.get_rgb_image_path(sample_name))
        image_shape = image.shape[0:2]

        #frame_calib = calib_utils.get_frame_calib(dataset.calib_dir, sample_name)
        #cam_p = frame_calib.p2

        # Get calibration
        calib = moose_load_calibration.load_calibration(dataset.calib_dir)

        img_idx = int(sample_name)

        if img_idx < 100:

            T_IMG_CAM = np.eye(4);  # identity matrix
            T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM00']['camera_matrix']['data']).reshape(-1,
                                                                                            3)  # camera to image #intrinsic matrix

            # T_IMG_CAM : 4 x 4 matrix
            T_IMG_CAM = T_IMG_CAM[0:3, 0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

            cam_p = T_IMG_CAM

        elif (img_idx >= 100 and img_idx < 200):
            T_IMG_CAM = np.eye(4);  # identity matrix
            T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM01']['camera_matrix']['data']).reshape(-1,
                                                                                            3)  # camera to image #intrinsic matrix

            # T_IMG_CAM : 4 x 4 matrix
            T_IMG_CAM = T_IMG_CAM[0:3, 0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

            cam_p = T_IMG_CAM

        elif (img_idx >= 200 and img_idx < 300):
            T_IMG_CAM = np.eye(4);  # identity matrix
            T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM02']['camera_matrix']['data']).reshape(-1,
                                                                                            3)  # camera to image #intrinsic matrix

            # T_IMG_CAM : 4 x 4 matrix
            T_IMG_CAM = T_IMG_CAM[0:3, 0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

            cam_p = T_IMG_CAM

        elif (img_idx >= 300 and img_idx < 400):
            T_IMG_CAM = np.eye(4);  # identity matrix
            T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM03']['camera_matrix']['data']).reshape(-1,
                                                                                            3)  # camera to image #intrinsic matrix

            # T_IMG_CAM : 4 x 4 matrix
            T_IMG_CAM = T_IMG_CAM[0:3, 0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

            cam_p = T_IMG_CAM

        elif (img_idx >= 400 and img_idx < 500):
            T_IMG_CAM = np.eye(4);  # identity matrix
            T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM04']['camera_matrix']['data']).reshape(-1,
                                                                                            3)  # camera to image #intrinsic matrix

            # T_IMG_CAM : 4 x 4 matrix
            T_IMG_CAM = T_IMG_CAM[0:3, 0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

            cam_p = T_IMG_CAM

        elif (img_idx >= 500 and img_idx < 600):
            T_IMG_CAM = np.eye(4);  # identity matrix
            T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM05']['camera_matrix']['data']).reshape(-1,
                                                                                            3)  # camera to image #intrinsic matrix

            # T_IMG_CAM : 4 x 4 matrix
            T_IMG_CAM = T_IMG_CAM[0:3, 0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

            cam_p = T_IMG_CAM

        elif (img_idx >= 600 and img_idx < 700):
            T_IMG_CAM = np.eye(4);  # identity matrix
            T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM06']['camera_matrix']['data']).reshape(-1,
                                                                                            3)  # camera to image #intrinsic matrix

            # T_IMG_CAM : 4 x 4 matrix
            T_IMG_CAM = T_IMG_CAM[0:3, 0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

            cam_p = T_IMG_CAM

        elif (img_idx >= 700 and img_idx < 800):
            T_IMG_CAM = np.eye(4);  # identity matrix
            T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM07']['camera_matrix']['data']).reshape(-1,
                                                                                            3)  # camera to image #intrinsic matrix

            # T_IMG_CAM : 4 x 4 matrix
            T_IMG_CAM = T_IMG_CAM[0:3, 0:4];  # remove last row, #choose the first 3 rows and get rid of the last column

            cam_p = T_IMG_CAM

        else:
            print("YOLO")


        # Load point cloud
        point_cloud = dataset.get_point_cloud('lidar', int(sample_name), image_shape)

        # Fill depth map
        if fill_type == 'multiscale':
            # Project point cloud to depth map
            projected_depths = depth_map_utils.project_depths(point_cloud, cam_p, image_shape)

            start_fill_time = time.time()
            final_depth_map, _ = ip_depth_map_utils.fill_in_multiscale(projected_depths)
            end_fill_time = time.time()
        else:
            raise ValueError('Invalid fill algorithm')

        # Save depth maps
        if save_depth_maps:
            out_depth_map_path = out_depth_map_dir + '/{}.png'.format(sample_name)
            depth_map_utils.save_depth_map(out_depth_map_path, final_depth_map)

        # Stop timing
        end_total_time = time.time()

        # Update fill times
        last_fill_times = np.roll(last_fill_times, -1)
        last_fill_times[-1] = end_fill_time - start_fill_time

        # Update total times
        last_total_times = np.roll(last_total_times, -1)
        last_total_times[-1] = end_total_time - start_total_time


if __name__ == "__main__":
    main()