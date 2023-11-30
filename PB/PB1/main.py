import scipy.io
import numpy as np
import matplotlib.pyplot as plt

data = scipy.io.loadmat("image_and_base.mat")

base30 = np.array(data['base30'])
im = np.array(data['im'])
imb = np.array(data['imb'])
indxbase = np.array(data['indxbase'])

print(indxbase)
plt.imshow(indxbase)
plt.show()

