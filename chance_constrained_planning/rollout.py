import numpy as np


def rollout_nominal(start, path):
    traj = [start.copy()]
    for wp in path:
        traj.append(wp.copy())
    return traj


def rollout_optimized(start, path, obstacle_points, solver, model, device):
    p = start.copy()
    traj = [p.copy()]

    for wp in path:
        p = solver(p, wp, obstacle_points, model, device)
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