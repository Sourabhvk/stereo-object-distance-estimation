# Data Structure

This project uses KITTI stereo data. The dataset contains left and right camera images, calibration files, and extra ground-truth files that are useful later for evaluation.

## Main Project Data Folder

```text
data/
+-- KITTI_RAW DATA/
¦   +-- data_scene_flow/
¦   +-- data_scene_flow_calib/
+-- sample/
    +-- calib/
    +-- left/
    +-- right/
```

## Folders We Use First

`image_2` contains the left camera images.

`image_3` contains the right camera images.

A matching stereo pair uses the same filename in both folders:

```text
training/image_2/000000_10.png
training/image_3/000000_10.png
```

For the first stereo depth version, use `_10` images first and ignore `_11`.

## Filename Meaning

`000000_10.png` means scene/frame ID `000000`, time step `10`.

`000000_11.png` means the same scene at the next time step, `11`.

## Ground Truth Folders

`disp_occ_0` contains ground-truth disparity for `_10`, including occluded pixels.

`disp_occ_1` contains ground-truth disparity for `_11`, including occluded pixels.

`disp_noc_0` contains ground-truth disparity for `_10`, using non-occluded pixels only.

`disp_noc_1` contains ground-truth disparity for `_11`, using non-occluded pixels only.

These are not input images for the first pipeline. They are useful later to compare our computed disparity against KITTI ground truth.

## Folders To Ignore For Now

`flow_occ` and `flow_noc` are optical flow ground truth folders.

`viz_flow_occ` and `viz_flow_occ_dilate_1` are optical flow visualization folders.

`obj_map` contains object or motion masks.

These are useful for other computer vision tasks, but not needed for the first stereo distance estimation pipeline.

## Calibration Folder

```text
data_scene_flow_calib/training/calib_cam_to_cam/
```

This folder contains files like:

```text
000000.txt
```

These files store camera projection matrices. Later, we will parse them to get the focal length and stereo baseline for the depth formula:

```text
depth = focal_length * baseline / disparity
```

## Version 1 Scope

For the first version, use only:

```text
data/KITTI_RAW DATA/data_scene_flow/training/image_2
data/KITTI_RAW DATA/data_scene_flow/training/image_3
data/KITTI_RAW DATA/data_scene_flow_calib/training/calib_cam_to_cam
```
