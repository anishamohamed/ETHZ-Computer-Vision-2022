import numpy as np


def propagate(particles, frame_height, frame_width, params):
    # For noth (i) no motion model and (ii) motion model, calsculate:
    #   -   A: deterministic component, models system knowledge
    #   -   B: stochastic component, models uncertainties
    #   --> s'^n_{t} = A s^n_{t-1} + B w^n_{t-1}

    # (i) no motion model
    if params["model"] == 0:
        A = np.eye(2) 
        B = np.array([params["sigma_position"], params["sigma_position"]])

    # (ii) motion model
    if params["model"] == 1:
        dt = 1
        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])        
        B = np.array([params["sigma_position"], params["sigma_position"],
                          params["sigma_velocity"], params["sigma_velocity"]])

    deterministic = np.matmul(A, particles.T).T
    stochastic = B * np.random.randn(*particles.shape)

    res = deterministic + stochastic
    res[:, 0] = np.clip(res[:, 0], 0, frame_width - 1)
    res[:, 1] = np.clip(res[:, 1], 0, frame_height - 1)

    return res
