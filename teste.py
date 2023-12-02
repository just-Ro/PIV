import numpy as np
from scipy.io import savemat, loadmat
from sklearn.neighbors import NearestNeighbors
from pprint import pprint
import matplotlib.pyplot as plt


# Generate example keypoints
np.random.seed(42)  # for reproducibility
N_features1 = 100
N_features2 = 10

features1 = np.random.rand(130, N_features1).T
features2 = np.random.rand(130, N_features2).T


frame1 = np.array([[1,1,1,1],[1,-1,1,-1],[-1,-1,-1,-1],[-1,1,-1,1]])
frame2 = np.array([[-1.2,-1,-1.2,-1],[-1.2,1,-1.2,1],[1.2,1,1.2,1],[1.2,-1,1.2,-1],[0,0,0,0],[0,0,0,0],[1.1,1.1,1.1,1.1]])


teste = frame1[[3,0]]
print(teste)

flagi = False

if frame1.shape[0] < frame2.shape[0]:
    frame1, frame2 = frame2, frame1
    flagi = True

features1 = frame1[:,2:]
features2 = frame2[:,2:]

# Create a NearestNeighbors model
knn_model = NearestNeighbors(n_neighbors=1)
knn_model.fit(features1)

# Use kneighbors to find the nearest neighbors
_, indices = knn_model.kneighbors(features2, n_neighbors=1)

keypoints1 = frame1[:, :2][indices.flatten()]
keypoints2 = frame2[:, :2]

if flagi:
    keypoints1, keypoints2 = keypoints2, keypoints1 #what the heeeeeellll


# Display the pairs
print('Keypoint Pairs:')
print('Image 1\tImage 2')
for i in range(len(keypoints1)):
    print('({:.2f}, {:.2f})\t({:.2f}, {:.2f})'.format(keypoints1[i, 0], keypoints1[i, 1],
                                                       keypoints2[i, 0], keypoints2[i, 1]))

# Plotting for visualization
plt.scatter(features1[:, 0], features1[:, 1], label='Keypoints in Image 1', marker='o', color='blue')
plt.scatter(features2[:, 0], features2[:, 1], label='Keypoints in Image 2', marker='x', color='red')

# Plot lines between corresponding keypoints
for i in range(len(keypoints1)):
    plt.plot([keypoints1[i, 0], keypoints2[i, 0]], [keypoints1[i, 1], keypoints2[i, 1]], color='green')

plt.legend()
plt.title('Corresponding Keypoints between Images')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

pprint((keypoints1, keypoints2))
