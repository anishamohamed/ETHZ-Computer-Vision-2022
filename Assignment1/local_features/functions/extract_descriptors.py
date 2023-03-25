import numpy as np

def filter_keypoints(img, keypoints, patch_size=9):
    # TODO: Filter out keypoints that are too close to the edges
    img_h, img_w = img.shape
    d = patch_size // 2

    xcoords, ycoords = keypoints[:,0].flatten(), keypoints[:,1].flatten()
    filtered_xcoords_indices = np.logical_and(xcoords >= d, xcoords <= (img_w - d))
    filtered_ycoords_indices = np.logical_and(ycoords >= d, ycoords <= (img_h - d))
    
    # return keypoints[np.intersect1d(filtered_xcoords_indices, filtered_ycoords_indices)]
    return keypoints[filtered_xcoords_indices & filtered_ycoords_indices]
    


# The implementation of the patch extraction is already provided here
def extract_patches(img, keypoints, patch_size = 9):
    '''
    Extract local patches for each keypoint
    Inputs:
    - img:          (h, w) gray-scaled images
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    - patch_size:   size of each patch (with each keypoint as its center)
    Returns:
    - desc:         (q, patch_size * patch_size) numpy array. patch descriptors for each keypoint
    '''
    h, w = img.shape[0], img.shape[1]
    img = img.astype(float) / 255.0
    offset = int(np.floor(patch_size / 2.0))
    ranges = np.arange(-offset, offset + 1)
    desc = np.take(img, ranges[:, None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc = desc.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    return desc