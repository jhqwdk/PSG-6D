# PSG-6D:Prior-free Implicit Category-level 6D Pose Estimation with SO(3)-Equivariant Networks and Point Cloud Global Enhancement

## Getting startted

*Prepare the environment*
``` conda create -n psg6d python=3.6
conda activate psg6d
# The code is tested on pytorch1.10 & CUDA10.2, please choose the properate vesion of torch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch -c conda-forge
# Dependent packages
pip install gorilla-core==0.2.5.3
pip install gpustat==1.0.0
pip install opencv-python-headless
pip install matplotlib
pip install scipy
```

*Compiling*
``` # Clone this repo
git clone https://github.com/jhqwdk/PSG-6D.git
# Compile pointnet2
cd model/pointnet2
python setup.py install
```

*Prepare the datasets*                                                                                                                                                                                              
Following DPDN, please Download the folloing data [NOCS](https://github.com/hughw19/NOCS_CVPR2019) ([camera_train](http://download.cs.stanford.edu/orion/nocs/camera_train.zip), [camera_test](http://download.cs.stanford.edu/orion/nocs/camera_val25K.zip), [camera_composed_depths](http://download.cs.stanford.edu/orion/nocs/camera_composed_depth.zip), [real_train](http://download.cs.stanford.edu/orion/nocs/real_train.zip), [real_test](http://download.cs.stanford.edu/orion/nocs/real_test.zip), [ground truths](http://download.cs.stanford.edu/orion/nocs/gts.zip), and [mesh models](http://download.cs.stanford.edu/orion/nocs/obj_models.zip)) and segmentation results ([Link](https://drive.google.com/file/d/1hNmNRr7YRCgg-c_qdvaIzKEd2g4Kac3w/view)). For camera_train.pkl, camera_val.pkl, real_test.pkl, real_rain.pkl, please refer to this [Link](https://drive.google.com/file/d/1Nz7cwcQWO_In4K6jKN1-5pQ0orY4UV9x/view?pli=1). Then unzip them in data folder and arange them as follows:
``` data
├── CAMERA
│   ├── train
│   └── val
├── camera_full_depths
│   ├── train
│   └── val
├── Real
│   ├── train
│   └── test
├── gts
│   ├── val
│   └── real_test
├── obj_models
│   ├── train
│   ├── val
│   ├── real_train
│   ├── real_test
│   ├── camera_train.pkl
│   ├── camera_val.pkl
│   ├── real_train.pkl
│   └── real_test.pkl
├── segmentation_results
    ├── train_trainedwoMask
    ├── test_trainedwoMask
    └── test_trainedwithMask
```

*Date processing*
```
python data_processing.py
```

## Training from scartch
``` # gpus refers to the ids of gpu. For single gpu, please set it as 0
python train.py --gpus 0,1 --config config/psg6d_default.yaml
```

## Training in seperate manner
If you want to achieve a higher result, we recommand you to train PSG6D in two phase. Phase 1, train the world-space part. Phase 2, freeze the world-space and train other component from scartch.
```                                                                                                                                                                                                                
 # Phase 1
python train.py --gpus 0,1 --config config/posenet_gt_vnn_glo.yaml
# Phase 2, modify the [world_enhancer_path] in yaml file with the model weights saved in phase 1
python train.py --gpus 0,1 --config config/psg6d_freeze_world_enhancer.yaml
```

## Evaluation
The two phase weight can be found here: [weight](https://drive.google.com/drive/my-drive)
```
python test.py --config config/psg6d_freeze_world_enhancer.yaml
```



# Acknowledgement
* Our code is developed upon [DPDN](https://github.com/JiehongLin/Self-DPDN),[IST-Net](https://github.com/CVMI-Lab/IST-Net?tab=readme-ov-file#prepare-the-environment).
* The dataset is provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019).
