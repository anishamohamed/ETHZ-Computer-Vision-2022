import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost


def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):

    xmin = particles[:, 0] - bbox_width / 2
    xmax = particles[:, 0] + bbox_width / 2
    ymin = particles[:, 1] - bbox_height / 2
    ymax = particles[:, 1] + bbox_height / 2

    particles_w = np.zeros((particles.shape[0], 1))

    for i in range(len(particles)):
        hist_particle = color_histogram(xmin[i], ymin[i], xmax[i], ymax[i], frame, hist_bin)
        chi = chi2_cost(hist_particle, hist)
        particles_w[i] = (1 / (np.sqrt(2 * np.pi) * sigma_observe)) * \
                         np.exp(- (chi ** 2) / ((sigma_observe ** 2) * 2))

    particles_w = particles_w / np.sum(particles_w)

    return particles_w
