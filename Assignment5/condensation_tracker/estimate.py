import numpy as np

def estimate(particles, particles_w):
    weighted = particles * particles_w
    return np.sum(weighted, axis=0) 