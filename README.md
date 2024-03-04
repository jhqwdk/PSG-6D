# PSG-6D:Prior-free Implicit Category-level 6D Pose Estimation with SO(3)-Equivariant Networks and Point Cloud Global Enhancement

## Getting startted

*Prepare the environment*
···
conda create -n istnet python=3.6
conda activate istnet
# The code is tested on pytorch1.10 & CUDA11.3, please choose the properate vesion of torch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# Dependent packages
pip install gorilla-core==0.2.5.3
pip install gpustat==1.0.0
pip install opencv-python-headless
pip install matplotlib
pip install scipy
