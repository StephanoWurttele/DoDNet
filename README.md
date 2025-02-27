# DoDNet
<p align="left">
    <img src="a_DynConv/dodnet.png" width="85%" height="85%">
</p>


This repo holds the pytorch implementation of DoDNet:<br />

**DoDNet: Learning to segment multi-organ and tumors from multiple partially labeled datasets.** 
(https://arxiv.org/pdf/2011.10217.pdf)

## Requirements
Python 3.7<br />
PyTorch==1.4.0<br />
[Apex==0.1](https://github.com/NVIDIA/apex)<br />
[batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)<br />

## Usage

### 0. Installation
* Clone this repo
```
git clone https://github.com/jianpengz/DoDNet.git
cd DoDNet
```
### 0.5 Installation packages
Use Python Version 3.7.13.

If using conda, run the following commands:
```
conda create --name DoDNet python=3.7.13
conda activate DoDNet
pip install batchgenerators==0.20
conda install -c anaconda pillow==7.0.0
conda install -c pytorch pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1
conda install -c conda-forge opencv
conda install -c conda-forge matplotlib==3.2.2
conda install -c conda-forge nibabel
conda install -c simpleitk simpleitk
conda install -c conda-forge tensorboardx
```
For Apex do:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```
On windows:
For Apex, a requirement might be to install Miscrosoft Visual C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/. Also, must revert to commit [#793](https://github.com/NVIDIA/apex/commit/2ec84ebdca59278eaf15e8ddf32476d9d6d8b904).

If you get a "cuda version" or related error: In setup.py, in the method check_cuda_torch_binary_vs_bare_metal, comment the if with the Raise Error in case your CUDA version mismatches. Might raise errors, but also may not.

### 1. MOTS Dataset Preparation
Before starting, MOTS should be re-built from the serveral medical organ and tumor segmentation datasets

Partial-label task | Data source
--- | :---:
Liver | [data](https://competitions.codalab.org/competitions/17094)
Kidney | [data](https://kits19.grand-challenge.org/data/)
Hepatic Vessel | [data](http://medicaldecathlon.com/)
Pancreas | [data](http://medicaldecathlon.com/)
Colon | [data](http://medicaldecathlon.com/)
Lung | [data](http://medicaldecathlon.com/)
Spleen | [data](http://medicaldecathlon.com/)

```
pip install skimage   / conda install -c anaconda scikit-image
pip install tqdm      / conda install -c conda-forge tqdm
pip install SimpleITK / conda install -c simpleitk simpleitk

```

* Download and put these datasets in `dataset/0123456/`. 
* Re-spacing the data by `python re_spacing.py`, the re-spaced data will be saved in `0123456_spacing_same/`.

The folder structure of dataset should be like

    dataset/0123456_spacing_same/
    ├── 0Liver
    |    └── imagesTr
    |        ├── liver_0.nii.gz
    |        ├── liver_1.nii.gz
    |        ├── ...
    |    └── labelsTr
    |        ├── liver_0.nii.gz
    |        ├── liver_1.nii.gz
    |        ├── ...
    ├── 1Kidney
    ├── ...


### 2. Model
Pretrained model is available in [checkpoint](https://drive.google.com/file/d/1qj8dJ_G1sHiCmJx_IQjACQhjUQnb4flg/view?usp=sharing). Use this to skip step 3.

### 3. Training
This step can be skipped and go straight to step 4 if Model from step 2 was installed.


Install dependencies:
```
!pip install SimpleITK
!pip install batchgenerators
!pip install batchgenerators==0.20.0
!pip install tensorboardX
```

If you require to install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir \
--global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

If you get a "cuda version" or related error: In `setup.py`, in the method `check_cuda_torch_binary_vs_bare_metal`, comment the if with the Raise Error in case your CUDA version mismatches. Might raise errors, but also may not.


* cd `a_DynConv/' and run 
```
!CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM ./a_DynConv/train.py \
--train_list='list/MOTS/MOTS_train_con_pancreas.txt' \
--snapshot_dir='snapshots/dodnet' \
--input_size='64,192,192' \
--batch_size=2 \
--num_gpus=1 \
--num_epochs=1000 \
--start_epoch=0 \
--learning_rate=1e-2 \
--num_classes=2 \
--num_workers=8 \
--weight_std=True \
--random_mirror=True \
--random_scale=True \
--FP16=False
```

### 4. Evaluation
```
CUDA_VISIBLE_DEVICES=0 python ./a_DynConv/evaluate.py \
--val_list='list/MOTS/MOTS_test.txt' \
--reload_from_checkpoint=True \
--reload_path='snapshots/dodnet/MOTS_DynConv_checkpoint.pth' \
--save_path='outputs/' \
--input_size='64,192,192' \
--batch_size=1 \
--num_gpus=1 \
--num_workers=2
```

In order to visualize using "revealSegmentation.py", consider the line 195 "np.save("pred_organ.npy",pred_organ)" which was not on the original code. With this, revealSegmentation can have something to work with.

# 4.1. Private Evaluation

For Hospital Guillermo Almenara CTs, when acquired, include dicom files in `/dataset/patient/dicom_files`. Then run `a_DynConv/niiGeneration`. If in windows, might need to replace line to:
```
dicom2nifti.convert_directory("..\\dataset\\patient\\dicom_files","..\\dataset\\patient")
```
Then evaluate by defining environment variable CUDA_VISIBLE_DEVICES = 0 with
```
setx CUDA_VISIBLE_DEVICES 0
```
and restart your terminal. Then run
```
python a_DynConv/evaluate_patient.py --val_list=list/MOTS/MOTS_test.txt --reload_from_checkpoint=True --reload_path=./snapshot/dodnet/MOTS_DynConv_checkpoint_v1.pth --save_path=output --input_size=64,192,192 --batch_size=1 --num_gpus=1 --num_workers=2
```

Then run `re_spacing_patient.py` to prepare input. (I think this is not necessary)

### 5. Post-processing
En el archivo postp.py, remover parametro neighbor en las funciones LAB() (linea 36 y 54)

```
python postp.py --img_folder_path='outputs/dodnet/'
```

To visualize the segmentation, run

```
python revealSegmentation.py
```

### 6. Citation
If this code is helpful for your study, please cite:
```
@inproceedings{zhang2021dodnet,
  title={DoDNet: Learning to segment multi-organ and tumors from multiple partially labeled datasets},
  author={Zhang, Jianpeng and Xie, Yutong and Xia, Yong and Shen, Chunhua},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={},
  year={2021}
}
```


### Contact
Jianpeng Zhang (james.zhang@mail.nwpu.edu.cn)
