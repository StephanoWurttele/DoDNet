from PIL import Image
import numpy as np

data = np.load("/output/pred_organ.npy")
print(data.shape)
print(data.sum())
print(data[:,:,90].sum())

img = Image.fromarray((data[:,:,90]*255).astype('uint8'), mode="L")
img.save('output/ejemplo-p11.jpg')
img.show() 