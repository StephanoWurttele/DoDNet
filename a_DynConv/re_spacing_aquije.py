import numpy as np
import os
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import SimpleITK as sitk

spacing = {
    0: [1.5, 0.8, 0.8],
    1: [1.5, 0.8, 0.8],
    2: [1.5, 0.8, 0.8],
    3: [1.5, 0.8, 0.8],
    4: [1.5, 0.8, 0.8],
    5: [1.5, 0.8, 0.8],
    6: [1.5, 0.8, 0.8],
}

count = -1

img_path = "/content/drive/MyDrive/501_arterial_idose_4.nii.gz"
print(img_path)
imageITK = sitk.ReadImage(img_path)
image = sitk.GetArrayFromImage(imageITK)
ori_size = np.array(imageITK.GetSize())[[2, 1, 0]]
ori_spacing = np.array(imageITK.GetSpacing())[[2, 1, 0]]
ori_origin = imageITK.GetOrigin()
ori_direction = imageITK.GetDirection()

task_id = 3
target_spacing = np.array(spacing[task_id])
spc_ratio = ori_spacing / target_spacing

data_type = image.dtype

data_type = np.int32
   
order = 3
mode_ = 'constant'

image = image.astype(np.float)

image_resize = resize(image, (int(ori_size[0] * spc_ratio[0]), int(ori_size[1] * spc_ratio[1]),
int(ori_size[2] * spc_ratio[2])),order=order, mode=mode_, cval=0, clip=True, preserve_range=True)
image_resize = np.round(image_resize).astype(data_type)

saveITK = sitk.GetImageFromArray(image_resize)
saveITK.SetSpacing(target_spacing[[2, 1, 0]])
saveITK.SetOrigin(ori_origin)
saveITK.SetDirection(ori_direction)
sitk.WriteImage(saveITK, os.path.join("/content/drive/MyDrive/respacing_aquije/","aquije_spacing.nii.gz"))
