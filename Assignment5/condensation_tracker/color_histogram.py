import numpy as np
from cv2 import calcHist

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    particle = frame[
        max(0, round(ymin)): min(frame.shape[0]-1, round(ymax)), 
        max(0, round(xmin)): min(frame.shape[1]-1, round(xmax))
    ]
    hist = np.empty((3, hist_bin))
    for channel in range(frame.shape[-1]):
        hist[channel], _ = np.histogram(particle[:, :, channel], bins=hist_bin)

    # normalize histogram
    hist = hist / hist.sum() 
    hist = hist.flatten()

    return hist