import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the second image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please

    desc1 = np.reshape(desc1, (-1, 1, desc1.shape[-1]))
    desc2 = np.reshape(desc2, (1, -1, desc2.shape[-1]))   

    distances = np.sum((desc1 - desc2) ** 2, axis=-1)
    return distances

def match_descriptors(desc1, desc2, method="one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the second image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None

    nearest_neighbour = distances == np.min(distances, axis=-1).reshape(-1, 1)

    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest matching here
        matches = np.argwhere(nearest_neighbour)

    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # least_distant_other_dir = np.argmin(distances, axis=0) # indices of q1
        # matches = np.array([
        #     [least_distant_other_dir[i], i] for i in range(q2)
        #     if least_distant[least_distant_other_dir[i]] == i
        # ])
        nearest_neighbour_opposite = distances == np.min(distances, axis=-2).reshape(1, -1)
        matches = np.argwhere(np.logical_and(nearest_neighbour, nearest_neighbour_opposite))

    elif method == "ratio":
        # TODO: implement the ratio test matching here
        two_least_distant = np.partition(distances, 2)[:,:2]
        ratio_check = two_least_distant[:,0] / two_least_distant[:,1] < ratio_thresh
        matches = np.argwhere(nearest_neighbour)[ratio_check]
    else:
        raise NotImplementedError
    return matches