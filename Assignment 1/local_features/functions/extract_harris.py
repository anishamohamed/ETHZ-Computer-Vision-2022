import numpy as np
import cv2 as cv

from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter

IMG_NAME1 = "images/blocks.jpg"
IMG_NAME2 = "images/house.jpg"

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0
 
    # 1.1 Compute image gradients
    # TODO: implement the computation of the image gradients Ix and Iy here.

    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.

    Ix = convolve2d(img, np.array([1, 0, -1]).reshape(1, -1) * 0.5, mode="same")
    Iy = convolve2d(img, np.array([1, 0, -1]).reshape(-1, 1) * 0.5, mode="same")

    # 1.2 Compute local auto-correlation matrix
    # TODO: compute the auto-correlation matrix here
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)

    # 1.2.1 Compute products of derivatives at every pixel
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # 1.2.2 Compute the sums of the products of derivatives at each pixel
    Sxx = cv.GaussianBlur(Ixx, (5, 5), sigma, borderType=cv.BORDER_REPLICATE) # (h, w)
    Syy = cv.GaussianBlur(Iyy, (5, 5), sigma, borderType=cv.BORDER_REPLICATE) # (h, w)
    Sxy = cv.GaussianBlur(Ixy, (5, 5), sigma, borderType=cv.BORDER_REPLICATE) # (h, w)

    # Define the matrix M(x,y)=[[S_x2,S_xy],[S_xy,S_y2]]
    M = np.array([[Sxx,Sxy],[Sxy,Syy]]) # (2, 2, h, w)

    # 1.3 Compute Harris response function
    # TODO: compute the Harris response function C here

    # C(i, j) = det(M(i,j) - k * trace(M(i,j)))^2
    det = Sxx * Syy - Sxy ** 2 # (h, w)
    trace = Sxx + Syy # (h, w)
        
    C = det - k * (trace ** 2)

    # Detection with threshold
    # TODO: detection and find the corners here
    # For the local maximum check, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    
    # 1.3.1 Check condition 1: C(i, j) > thresh
    condition1 = C > thresh

    # 1.3.2 Check condition 2: maximum in 3x3 neighbourhood
    condition2 = (C == maximum_filter(C, 3))

    corners = np.argwhere((condition1 & condition2))
    corners[:,[0,1]] = corners[:, [1,0]] # draw_keypoints switches between x and y coord

    return corners, C