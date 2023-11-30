from scipy import ndimage
import numpy as np
import cv2 as cv

def padding(image, padding_size: int, color: int=255):

    # Create a larger canvas with padding
    canvas = np.ones((image.shape[0] + 2 * padding_size, image.shape[1] + 2 * padding_size, 3), dtype=np.uint8) * color

    # Place the image with padding on the canvas
    canvas[padding_size:padding_size + image.shape[0], padding_size:padding_size + image.shape[1]] = image
    
    return canvas

def findHomography(src_pts, dst_pts):
    # Find homography matrix using Direct Linear Transform (DLT)
    if len(src_pts) != len(dst_pts):
        raise ValueError("There must be the same number of source and destination points")
    if len(src_pts) < 4:
        raise ValueError("More than 4 points are required")
    
    A = []
    for src, dst in zip(src_pts, dst_pts):
        x, y = src
        u, v = dst
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)
    
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    H /= H[2, 2]
    
    return H
def bilinear_interpolation(img, x, y):
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1

    if x1 >= img.shape[1]:
        x1 = x0
    if y1 >= img.shape[0]:
        y1 = y0

    Q11 = img[y0, x0]
    Q21 = img[y0, x1]
    Q12 = img[y1, x0]
    Q22 = img[y1, x1]

    x_weight = x - x0
    y_weight = y - y0

    top_interp = Q21 * x_weight + Q11 * (1 - x_weight)
    bottom_interp = Q22 * x_weight + Q12 * (1 - x_weight)

    interpolated_value = bottom_interp * y_weight + top_interp * (1 - y_weight)

    return interpolated_value
    
def warpPerspective1(src, H, dst_size):
    # Generate an empty image
    width, height = dst_size
    channel = src.shape[2] if src.ndim > 2 else 1
    dst = np.zeros((height, width, channel), dtype=src.dtype)
    
    # Iterate over the destination image
    for qy in range(height):
        for qx in range(width):
            # Calculate the inverse mapping using the homography matrix H
            q = np.dot(np.linalg.inv(H), [qx, qy, 1])
            px, py = int(q[0]/q[-1] + 0.5), int(q[1]/q[-1] + 0.5)
            
            # Check if the source pixel is within bounds
            if px >= 0 and py >= 0 and px < src.shape[1] and py < src.shape[0]:
                dst[qy, qx] = src[py, px]
    
    return dst

def warpPerspective(image, homography_matrix, output_shape):

    # Create a meshgrid of coordinates for the new perspective
    x, y = np.meshgrid(np.arange(output_shape[1]), np.arange(output_shape[0]))
    coordinates = np.vstack((x.ravel(), y.ravel(), np.ones_like(x.ravel())))

    # Apply the homography matrix to the coordinates
    transformed_coordinates = np.dot(homography_matrix, coordinates)
    transformed_coordinates /= transformed_coordinates[2, :]

    # Use map_coordinates for image warping
    warped_image = ndimage.map_coordinates(image, transformed_coordinates[:2, :].reshape(2, *output_shape), order=1, mode='constant').reshape(output_shape)

    return warped_image.astype(np.uint8)

border = 100

im1 = padding(cv.imread('PB\PB5\parede1.jpg'),border,0)
im2 = padding(cv.imread('PB\PB5\parede2.jpg'),border,0)


c1 = np.array([[67,21],[156,32],[62,168],[160,170]], dtype=np.float32) + border
c2 = np.array([[135,54],[213,50],[125,186],[206,196]], dtype=np.float32) + border
world = np.array([[0,0],[841,0],[0,1189],[841,1189]])*0.09 + 2*border

H1 = findHomography(c1, world)
H2 = findHomography(c2, world)

# Warp both images
result_image1 = warpPerspective1(im1, H1, (im1.shape[1], im1.shape[0]))
result_image2 = warpPerspective1(im2, H2, (im2.shape[1], im2.shape[0]))

blended_result = cv.addWeighted(result_image1, 0.5, result_image2, 0.5, 0)

for point in c1:
    cv.circle(im1, tuple(map(int, point)), 5, (0, 255, 0), -1)

for point in c2:
    cv.circle(im2, tuple(map(int, point)), 5, (0, 255, 0), -1)

cv.imshow('Image 1', im1)
cv.imshow('Image 2', im2)
cv.imshow('Warped Image 1', result_image1)
cv.imshow('Warped Image 2', result_image2)
cv.imshow('Blended Result', blended_result)
cv.waitKey(0)
cv.destroyAllWindows()