import numpy as np
import torch

def rollout_nominal(start, path):
    traj = [start.copy()]
    for wp in path:
        traj.append(wp.copy())
    return traj

def _to_numpy(x):
    # torch Tensor → numpy (CPU)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    # already numpy / list / tuple → numpy
    return np.asarray(x)

def rollout_optimized(start, path, obstacle_points, solver, model, device):
    p = _to_numpy(start).copy()
    traj = [p.copy()]

    for wp in path:
        wp_np = _to_numpy(wp)
        p, mu, sigma = solver(p, wp_np, obstacle_points, model, device)
        p = _to_numpy(p).copy()
        traj.append(p.copy())

    return traj

def rollout_optimized_plot(start, path, obstacle_points, solver, model, device):
    p = start.copy()

    traj = [p.copy()]
    mus = []
    sigmas = []

    for wp in path:
        p, mu, sigma = solver(p, wp, obstacle_points, model, device)

        traj.append(p.copy())
        mus.append(mu)
        sigmas.append(sigma)

    return traj, mus, sigmas