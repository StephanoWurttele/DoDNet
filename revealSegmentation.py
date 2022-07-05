from PIL import Image
import numpy as np
import os
original_input = np.load("./dataset/patient/input_npys/input_image0.npy")
original_input = original_input[0][0]
BandW_pred = np.load("./dataset/patient/results/patient_prediccion2.npy")

BandW_pred = np.moveaxis(BandW_pred, 0, -1)
BandW_pred = np.moveaxis(BandW_pred, 0, -1)
print(original_input.shape)
print(BandW_pred.shape)
npy_with_masking=np.multiply(original_input,BandW_pred)

path_base=os.getcwd()
save_path=os.path.join(path_base,"dataset/patient/results/visibleSeg")
it = 0
for shape in npy_with_masking:
    name = str(it) + ".npy"
    filename=os.path.join(save_path,name)
    np.save(filename, shape)
    it += 1
    print("File " + filename + " generated.")

num = input("Enter npy number")
original_input = np.load(save_path+"/"+num+".npy")

img = Image.fromarray((original_input * 255).astype('uint8') , 'L')

### SHOW IN B&W
# print(data.shape)
# print(data.sum())
# print(data[:,:,68].sum())

# img = Image.fromarray((data[:,:,68]*255).astype('uint8'), mode="L")
img.save('./dataset/patient/results/finalseg68.jpg')
img.show() 