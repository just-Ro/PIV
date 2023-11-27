# import numpy as np
# from scipy.io import savemat
# import scipy.io as sio

# f = sio.loadmat('output.mat')
# feat=f['features']
# print(type(feat))
# print(feat.shape)
# print(feat[0,2].shape)


import numpy as np
from scipy.io import savemat

# Create a list of NumPy arrays with variable sizes
array_list = [
    np.array([1, 2, 3]),
    np.array([[4, 5], [6, 7]]),
    np.array([[8, 9, 10], [11, 12, 13], [14, 15, 16]])
]

final = []
for a in array_list:
    final.append(a)

# Create a structured NumPy array with a field named 'arrays'
structured_array = np.array(final, dtype='object')

# Create a dictionary with the structured array under the key 'data'
data_dict = {'data': structured_array}



# Save the dictionary using savemat
savemat('output.mat', data_dict)
print('Done')
