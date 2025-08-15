import numpy as np

def add_noise(points, noise_std=0.01):
    noise = np.random.normal(scale=noise_std, size=points.shape)
    noisy_points = points + noise
    mask = np.zeros(len(points), dtype=bool)  # global noise usually not masked
    return noisy_points, mask
