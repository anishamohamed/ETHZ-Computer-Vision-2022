import numpy as np


def resample(particles, particles_w):

    indices = np.random.choice(
        len(particles),
        size=len(particles),
        replace=True,
        p=particles_w.flatten()
    )

    particles_update = particles[indices]
    particles_w_update = particles_w[indices] / sum(particles_w[indices])

    return particles_update, particles_w_update

    # resampled_particles = np.empty_like(particles)
    # resampled_particles_w = np.empty_like(particles_w)
    
    # N = len(particles)
    # c = particles_w[0]
    # r = np.random.rand(1)/N
    
    # idx = 0
    # for n in range(N):
    #     U = r + ((n-1) / N)
    #     while (U > c):
    #         c += particles_w[idx]
    #         idx += 1
    #         idx %= N
    #     resampled_particles[n] = particles[idx]
    #     resampled_particles_w[n] = particles_w[idx]
    
    # resampled_particles_w = resampled_particles_w / np.sum(resampled_particles_w)
    # return resampled_particles, resampled_particles_w