## Data Preparation


### 1. ScanNet Dataset
Please follow the official guideance of [ScanNet](http://www.scan-net.org/ScanNet/) dataset to sign the agreement and send it to scannet@googlegroups.com. After receiving the response, you can download the data.

After downloading the data and process it using the official script, you will get the following directory structure:

```
./data
  ├── scannet
      ├── scene0011_00
          ├── scene0011_00_vh_clean_2.ply      <- GT mesh of the scene
          ├── frames
              ├── color                           <- folder with RGB images
              │      ├── 0.jpg  (or .png/.jpeg)
              │      ├── 1.jpg (or .png/.jpeg)
              │      └── ...  
              ├── depth                           <- folder with depth images
              │      ├── 0.png  (or .jpg/.jpeg)
              │      ├── 1.png (or .jpg/.jpeg)
              │      └── ...  
              ├── intrinsic                 
              │      └── intrinsic_depth.txt      <- camera intrinsics
              |      └── ...
              └── pose                            <- folder with camera poses
                     ├── 0.txt
                     ├── 1.txt
                     └── ...  
     ├── sceneXXXX_XX
     └── ...
```


### 2. SceneNN Dataset
#### (1) Access and download raw data

You can access SceneNN dataset from the [official repository](https://github.com/hkust-vgd/scenenn). The IDs of sequences we use includes: [005, 011, 015, 030, 054, 080, 089, 093, 096, 243, 263, 322]. You can obtain the raw RGB-D data of selected sequences following the official guidance, or download them directly [here](https://hkust-vgd.ust.hk/scenenn/main/oni/). Then you need to download GT trajectory and mesh of selected sequences [here](https://drive.google.com/drive/folders/0B2BQi-ql8CzeMUZ4RUpnLW1JN3c?resourcekey=0-Ph3VntNntNqJ_CtSLnN7wA).

According to the [official guidance](https://github.com/hkust-vgd/scenenn), you will firstly get the following directory structure in the downloaded directory:

```
SceneNN
  ├── 005
  │   ├── 005.ply                 /* the reconstructed triangle mesh  */
  │   ├── 005.xml                 /* the annotation                   */
  │   ├── trajectory.log          /* camera pose (local to world)     */
  └── oni
      └── 005.oni                 /* the raw RGB-D video               */
  └── intrinsic
      └── asus.ini                /* intrinsic matrix data for Asus Xtion camera  */
  	└── kinect2.ini             /* intrinsic matrix data for Kinect v2 camera   */
```

#### (2) Extract and process
After downloading the selected sequences, you will get a file named `XXX.oni` for each sequence (e.g. `005.oni`). To extract raw RGB-D data for each scene, you can either refer to the [official repository](https://github.com/hkust-vgd/scenenn), or simply follow the steps below:

Step 1: clone the [SceneNN repository](https://github.com/hkust-vgd/scenenn), and compile it to get an executable file named `playback`:
```
sudo apt-get install libopenni-dev libfreeimage-dev libglew-dev freeglut3-dev
git clone https://github.com/hkust-vgd/scenenn.git
cd scenenn
cd playback && make all
```

Step 2: use the executable `playback` to extract a selected sequence:

Take `005.oni` as example, run the following command (You can modify the output directory as `<OnlineAnySeg_root>/data/SceneNN`):
```
./playback XXXX/005.oni XXXX/005
```
or you can process all selected sequences using this [script](../data/SceneNN/decompress_all_seqs.sh).


Step 3:
Link or copy the processed data sequences from downloaded directory to `<OnlineAnySeg_root>/data/SceneNN`, and you will get the following directory structure:

```
./data
  ├── SceneNN
      ├── 005
          ├── 005.ply                         <- GT mesh of the scene
          ├── trajectory.log                  <- GT trajectory of the scene (to be processed)
          ├── depth                           <- folder with depth images
          │      ├── depth00001.png
          │      ├── depth00002.png
          │      └── ...  
          └── image                           <- folder with RGB images
                 ├── image00001.png
                 ├── image00002.png
                 └── ...  
      ├── 011
      └── ...                                 <- other sequences
```

Step 4: get GT poses and camera intrinsic file:

run the [following script](../preprocess/SceneNN/process_gt_poses.py) to copy the intrinsic file to the directory of each scene, and get the GT poses in the target format:
```
python ./preprocess/SceneNN/process_gt_poses.py
```
and you will get the following structure finally:
```
./data
  ├── SceneNN
      ├── 005
          ├── 005.ply                         <- GT mesh of the scene
          ├── trajectory.log                  <- GT trajectory of the scene (to be processed)
          ├── depth                           <- folder with depth images
          │      ├── depth00001.png
          │      ├── depth00002.png
          │      └── ...  
          ├── image                           <- folder with RGB images
          │      ├── image00001.png
          │      ├── image00002.png
          │      └── ...  
          ├── intrinsic                 
          │      └── intrinsic_depth.txt      <- camera intrinsics
          └── pose                            <- folder with camera poses
                 ├── 1.txt
                 ├── 2.txt
                 └── ...  
      ├── 011
      └── ...                                 <- other sequences
```
Once you get the directory structure above under `<OnlineAnySeg_root>`, the data preparation for SceneNN dataset is finished.


### 3. Custom Sequences
For custom data sequences, it is recommended to adopt the following directory structure:
```
./data
  ├── My_dataset
      ├── My_sequence1
          ├── intrinsic_depth.txt             <- camera intrinsics
          ├── color                           <- folder with RGB images
          │      ├── 0.png  (or .jpg/.jpeg)
          │      ├── 1.png  (or .jpg/.jpeg)
          │      └── ...  
          ├── depth                           <- folder with depth images
          │      ├── 0.png  (or .jpg/.jpeg)
          │      ├── 1.png  (or .jpg/.jpeg)
          │      └── ...  
          └── poses                           <- folder with camera poses (4X4 camera_to_world matrix)
                 ├── 0.txt
                 ├── 1.txt
                 └── ...  
     ├── My_sequence2
     └── ...
```
